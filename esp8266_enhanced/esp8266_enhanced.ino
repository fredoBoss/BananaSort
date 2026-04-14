// esp8266_enhanced.ino
// Enhanced ESP8266 code for banana sorter weight node
// Changes vs original:
//   • Dynamic BASE_TARE_WEIGHT stored in EEPROM (set via serial command "settare")
//   • Uploads both raw and adjusted weight to Firebase for debugging
//   • Median-of-7 instead of median-of-5 for better noise rejection
//   • Publishes status flag ("ready" / "taring") so Python knows when to trust weight
//   • "forcetare" command re-tares and saves new base
//   • Reduced SYNC_INTERVAL to 300ms for more responsive reads

#include <ESP8266WiFi.h>
#include <Firebase_ESP_Client.h>
#include <EEPROM.h>
#include "HX711.h"

// ─── WiFi Credentials ──────────────────────────────
#define WIFI_SSID     "Dongpal 2G"
#define WIFI_PASSWORD "Bayuds2024"

// ─── Firebase Project Credentials ──────────────────
#define API_KEY      "AIzaSyA8Ln4PMiKmk7msvyM6iKGUbNXFs_h19U8"
#define DATABASE_URL "https://gradifier-aee7a-default-rtdb.asia-southeast1.firebasedatabase.app/"

// ─── Load Cell Pins ────────────────────────────────
#define LOADCELL_DOUT_PIN D4   // GPIO2
#define LOADCELL_SCK_PIN  D5   // GPIO14

// ─── EEPROM layout (bytes) ─────────────────────────
//  0..3   ZERO_OFFSET  (float)
//  4..7   SCALE_FACTOR (float)
//  8..11  balancePt[0] (float)
// 12..15  balancePt[1]
// 16..19  balancePt[2]
// 20..23  balancePt[3]
// 24..27  balancePt[4]
// 28..31  BASE_TARE_WEIGHT (float)   ← NEW

#define EEPROM_BASE_TARE_ADDR 28

// ─── Weight / calibration ─────────────────────────
float balancePt[5]   = {0,0,0,0,0};
int   currentPlate   = 1;
float ZERO_OFFSET    = 0.0f;
float SCALE_FACTOR   = 1.0f;
float BASE_TARE_WEIGHT = 912.88f;   // overwritten from EEPROM on boot
bool  calibrated     = false;

HX711 scale;

// ─── Firebase Objects ─────────────────────────────
FirebaseData  fbdo;
FirebaseAuth  auth;
FirebaseConfig config;

unsigned long lastSyncTime = 0;
const unsigned long SYNC_INTERVAL = 300;   // ms — faster than original 500ms

bool isTaring = false;   // status flag uploaded to Firebase

// ─────────────────────────────────────────────────────
// SETUP & LOOP
// ─────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\nESP8266 Enhanced Weight Node");

  EEPROM.begin(512);
  delay(300);

  connectWiFi();

  config.api_key      = API_KEY;
  config.database_url = DATABASE_URL;
  if (Firebase.signUp(&config, &auth, "", "")) {
    Serial.println("Firebase signUp OK");
  } else {
    Serial.println("Firebase signUp failed: " + String(config.signer.signupError.message.c_str()));
  }
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  initLoadCell();
  Serial.println("Setup complete. Type 'help' for commands.");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi lost — reconnecting…");
    connectWiFi();
  }

  if (millis() - lastSyncTime >= SYNC_INTERVAL) {
    lastSyncTime = millis();
    syncDataToFirebase();
  }

  if (Serial.available()) handleSerialCommand();

  delay(50);
  yield();
}

// ─────────────────────────────────────────────────────
// WIFI
// ─────────────────────────────────────────────────────
void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi OK  IP: " + WiFi.localIP().toString());
  } else {
    Serial.println("\nWiFi FAILED — will retry in loop");
  }
}

// ─────────────────────────────────────────────────────
// EEPROM
// ─────────────────────────────────────────────────────
void saveBaseTare(float value) {
  EEPROM.put(EEPROM_BASE_TARE_ADDR, value);
  EEPROM.commit();
  BASE_TARE_WEIGHT = value;
  Serial.println("BASE_TARE_WEIGHT saved: " + String(value, 2) + "g");
}

void loadBaseTare() {
  float v;
  EEPROM.get(EEPROM_BASE_TARE_ADDR, v);
  if (!isnan(v) && v > 0 && v < 5000) {
    BASE_TARE_WEIGHT = v;
    Serial.println("BASE_TARE loaded: " + String(v, 2) + "g");
  } else {
    Serial.println("No valid BASE_TARE in EEPROM — using default " + String(BASE_TARE_WEIGHT, 2) + "g");
  }
}

void clearEEPROM() {
  for (int i = 0; i < 512; i++) EEPROM.write(i, 0);
  EEPROM.commit();
  ZERO_OFFSET = 0; SCALE_FACTOR = 1; calibrated = false;
  Serial.println("EEPROM cleared");
}

void saveCalibration(float zo, float sf) {
  if (isnan(zo)||isnan(sf)||sf>0.02f||sf<0.00005f) {
    Serial.println("Invalid cal values — not saved");
    return;
  }
  EEPROM.put(0, zo);
  EEPROM.put(sizeof(float), sf);
  EEPROM.commit();
  ZERO_OFFSET  = zo;
  SCALE_FACTOR = sf;
  calibrated   = true;
  Serial.println("Cal saved  ZO=" + String(zo,2) + "  SF=" + String(sf,6));
}

void loadCalibration() {
  EEPROM.get(0, ZERO_OFFSET);
  EEPROM.get(sizeof(float), SCALE_FACTOR);
  if (isnan(ZERO_OFFSET)||isnan(SCALE_FACTOR)||
      ZERO_OFFSET==0||SCALE_FACTOR==0||
      SCALE_FACTOR>0.02f||SCALE_FACTOR<0.00005f) {
    ZERO_OFFSET=0; SCALE_FACTOR=1; calibrated=false;
    Serial.println("No valid cal in EEPROM");
  } else {
    calibrated = true;
    Serial.println("Cal loaded  ZO=" + String(ZERO_OFFSET,2) + "  SF=" + String(SCALE_FACTOR,6));
  }
}

void saveBalancePts() {
  int offset = sizeof(float)*2;
  for (int i=0;i<5;i++) EEPROM.put(offset+i*sizeof(float), balancePt[i]);
  EEPROM.commit();
}

void loadBalancePts() {
  int offset = sizeof(float)*2;
  for (int i=0;i<5;i++) {
    EEPROM.get(offset+i*sizeof(float), balancePt[i]);
    if (isnan(balancePt[i])) balancePt[i]=0;
  }
}

// ─────────────────────────────────────────────────────
// LOAD CELL
// ─────────────────────────────────────────────────────
void initLoadCell() {
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  int retries = 5;
  while (!scale.is_ready() && retries-->0) {
    Serial.println("HX711 not ready, retrying…");
    delay(1000);
  }
  if (!scale.is_ready()) {
    Serial.println("HX711 FAILED — halting");
    while(true) yield();
  }
  loadCalibration();
  loadBalancePts();
  loadBaseTare();           // ← NEW: load saved tare base
  scale.tare(20);
  balancePt[currentPlate-1] = weightValAvg();
  delay(500);
  Serial.println("Load cell ready");
}

float convertToWeight(long raw) {
  return (raw - ZERO_OFFSET) * SCALE_FACTOR;
}

// Median-of-7 (was median-of-5) — better outlier rejection
float weightValAvg() {
  delay(200);
  const int N = 7;
  long readings[N];
  for (int i=0;i<N;i++) {
    if (!scale.is_ready()) delay(50);
    readings[i] = scale.get_value();
    delay(5);
    yield();
  }
  // bubble sort
  for (int i=0;i<N-1;i++)
    for (int j=0;j<N-i-1;j++)
      if (readings[j]>readings[j+1]) {
        long t=readings[j]; readings[j]=readings[j+1]; readings[j+1]=t;
      }
  long raw = readings[N/2];   // median
  scale.power_down();
  delay(40);
  scale.power_up();
  return convertToWeight(raw);
}

// ─────────────────────────────────────────────────────
// FIREBASE SYNC
// ─────────────────────────────────────────────────────
void syncDataToFirebase() {
  if (WiFi.status() != WL_CONNECTED) return;
  if (isTaring) {
    Firebase.RTDB.setString(&fbdo, "Status", "taring");
    return;
  }

  float raw      = weightValAvg();
  float adjusted = raw - BASE_TARE_WEIGHT;
  if (adjusted < 0) adjusted = 0;

  // Upload adjusted weight (what Python reads)
  if (Firebase.RTDB.setFloat(&fbdo, "Weight", adjusted)) {
    Serial.println("Synced  raw=" + String(raw,1) + "g  adj=" + String(adjusted,1) + "g");
  } else {
    Serial.println("Firebase error: " + fbdo.errorReason());
  }

  // Upload raw for debug
  Firebase.RTDB.setFloat(&fbdo, "WeightRaw", raw);

  // Upload status
  Firebase.RTDB.setString(&fbdo, "Status", "ready");

  yield();
}

// ─────────────────────────────────────────────────────
// FORCE TARE  — re-tare with empty tray and save
// ─────────────────────────────────────────────────────
void forceTare() {
  Serial.println("Force tare — make sure tray is EMPTY then press Enter");
  unsigned long t = millis();
  while (!Serial.available() && millis()-t < 30000) {
    delay(100); yield();
  }
  if (!Serial.available()) { Serial.println("Timeout"); return; }
  Serial.readStringUntil('\n');

  isTaring = true;
  Serial.println("Taring…");
  scale.tare(25);
  delay(500);
  float raw = weightValAvg();
  saveBaseTare(raw);
  Serial.println("New BASE_TARE_WEIGHT = " + String(raw, 2) + "g");
  isTaring = false;
}

// ─────────────────────────────────────────────────────
// CALIBRATION
// ─────────────────────────────────────────────────────
bool waitY(unsigned long ms=60000) {
  Serial.println("Type 'y' + Enter to continue, anything else cancels…");
  unsigned long t = millis()+ms;
  while (millis()<t) {
    if (Serial.available()) {
      String s = Serial.readStringUntil('\n');
      s.trim(); s.toLowerCase();
      if (s=="y") return true;
      Serial.println("Cancelled"); return false;
    }
    delay(100); yield();
  }
  Serial.println("Timeout"); return false;
}

void calibrateWeight(int plate) {
  if (plate<1||plate>5) { Serial.println("Plate 1-5 only"); return; }
  Serial.println("\n=== CALIBRATE Plate " + String(plate) + " ===");

  Serial.println("Step 1: Empty plate — ready?");
  if (!waitY()) return;
  scale.tare(20);
  long zo = scale.read_average(10);
  Serial.println("Zero offset: " + String(zo));

  float known = 978.0;
  Serial.println("Step 2: Place " + String(known,0) + "g weight — ready?");
  if (!waitY()) return;
  long reading = scale.read_average(10);
  float diff   = reading - zo;
  if (diff==0) { Serial.println("No change — check wiring"); return; }
  float sf = known / diff;
  Serial.println("Scale factor: " + String(sf,6));
  if (sf>0.02f||sf<0.00005f) { Serial.println("Scale factor out of range"); return; }

  saveCalibration(zo, sf);

  Serial.println("Step 3: Remove weight — ready?");
  if (!waitY()) return;
  float bp = weightValAvg();
  balancePt[plate-1] = bp;
  saveBalancePts();
  currentPlate = plate;

  // auto-set tare base after calibration
  saveBaseTare(bp);

  Serial.println("Done! balancePt[" + String(plate) + "]=" + String(bp,2) + "g");
}

void printCalibration() {
  Serial.println("\n=== CALIBRATION INFO ===");
  Serial.println("ZO=" + String(ZERO_OFFSET,2) + "  SF=" + String(SCALE_FACTOR,6));
  Serial.println("calibrated=" + String(calibrated?"yes":"no"));
  Serial.println("BASE_TARE=" + String(BASE_TARE_WEIGHT,2) + "g");
  for (int i=0;i<5;i++)
    Serial.println("balancePt[" + String(i+1) + "]=" + String(balancePt[i],2));
}

// ─────────────────────────────────────────────────────
// SERIAL COMMANDS
// ─────────────────────────────────────────────────────
void handleSerialCommand() {
  String cmd = Serial.readStringUntil('\n');
  cmd.trim(); cmd.toLowerCase();

  if      (cmd=="cal1") calibrateWeight(1);
  else if (cmd=="cal2") calibrateWeight(2);
  else if (cmd=="cal3") calibrateWeight(3);
  else if (cmd=="cal4") calibrateWeight(4);
  else if (cmd=="cal5") calibrateWeight(5);
  else if (cmd=="info")      printCalibration();
  else if (cmd=="clear")     clearEEPROM();
  else if (cmd=="weight") {
    float w = weightValAvg() - BASE_TARE_WEIGHT;
    if (w<0) w=0;
    Serial.println("Weight: " + String(w,2) + "g");
  }
  else if (cmd=="sync")      syncDataToFirebase();
  else if (cmd=="forcetare") forceTare();   // ← NEW: re-tare and save base
  else if (cmd=="settare") {
    // Manual: settare:912.88
    // Usage: settare:<value>
    Serial.println("Use 'forcetare' to auto-tare with empty tray, or 'settare:<grams>'");
  }
  else if (cmd.startsWith("settare:")) {
    float v = cmd.substring(8).toFloat();
    if (v>0 && v<5000) saveBaseTare(v);
    else Serial.println("Invalid value");
  }
  else if (cmd.startsWith("plate")) {
    int p = cmd[5]-'0';
    if (p>=1&&p<=5) { currentPlate=p; Serial.println("Plate: " + String(p)); }
    else Serial.println("plate1-plate5");
  }
  else if (cmd=="diag") checkHX711Wiring();
  else if (cmd=="help") {
    Serial.println("\n=== COMMANDS ===");
    Serial.println("cal1-cal5   Calibrate plate");
    Serial.println("info        Show calibration");
    Serial.println("weight      Read weight");
    Serial.println("sync        Force Firebase sync");
    Serial.println("forcetare   Re-tare with empty tray + save");
    Serial.println("settare:<g> Manually set tare base");
    Serial.println("clear       Wipe EEPROM");
    Serial.println("diag        HX711 diagnostic");
    Serial.println("plate1-5    Set active plate");
  }
  else Serial.println("Unknown: '" + cmd + "' — type 'help'");
}

// ─────────────────────────────────────────────────────
// HX711 DIAGNOSTIC
// ─────────────────────────────────────────────────────
void checkHX711Wiring() {
  Serial.println("\n=== HX711 DIAGNOSTIC ===");

  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  delay(500);
  Serial.print("HX711 ready: ");
  Serial.println(scale.is_ready() ? "YES" : "NO — check wiring VCC/GND/DOUT/SCK");

  if (scale.is_ready()) {
    long r1=scale.read(); delay(100);
    long r2=scale.read(); delay(100);
    long r3=scale.read();
    Serial.println("Raw readings: " + String(r1) + ", " + String(r2) + ", " + String(r3));
    long var = max({r1,r2,r3}) - min({r1,r2,r3});
    Serial.println("Variance: " + String(var) + (var<50000?" (OK)":" (NOISY — check wires)"));
  }

  scale.power_down(); delay(100); scale.power_up(); delay(500);
  Serial.println("Power cycle: " + String(scale.is_ready()?"OK":"FAIL"));
  Serial.println("========================");
}
