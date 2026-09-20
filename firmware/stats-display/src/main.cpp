// dino-fly stats display — ESP32 DevKit + GMT130 1.3" ST7789 240x240 (no CS) + LED + reward/punish buttons + mode switch.
//
// Connects as a WebSocket client to the brain worker on the LAN (or to the public hub in cloud mode) and shows:
//   screen 0  big numbers: generation, best score, current score
//   screen 1  jumps / deaths / games this session, brain speed
//   screen 2  learning-curve sparkline (held-out mean per generation)
//   screen 3  live mini game: the fly's dino and the obstacles, straight from the canonical state in fly.frame
// LED flashes on every Giant Fiber spike. Buttons send `dopamine.event` (source = hardware). Long-press PUNISH = next screen.
//
// NOT YET TESTED ON HARDWARE (written against the documented wiring; see docs/HARDWARE.md).

#include <Arduino.h>
#include <ArduinoJson.h>
#include <WebSocketsClient.h>
#include <WiFi.h>

#define LGFX_USE_V1
#include <LovyanGFX.hpp>

#include "pins.h"
#include "secrets.h"

class Display : public lgfx::LGFX_Device {
  lgfx::Panel_ST7789 panel;
  lgfx::Bus_SPI bus;
  lgfx::Light_PWM light;

 public:
  Display() {
    auto b = bus.config();
    b.spi_host = VSPI_HOST;
    b.spi_mode = 3;  // GMT130 without CS needs SPI mode 3
    b.freq_write = 40000000;
    b.pin_sclk = PIN_TFT_SCK;
    b.pin_mosi = PIN_TFT_SDA;
    b.pin_miso = -1;
    b.pin_dc = PIN_TFT_DC;
    bus.config(b);
    panel.setBus(&bus);
    auto p = panel.config();
    p.pin_cs = -1;
    p.pin_rst = PIN_TFT_RES;
    p.panel_width = 240;
    p.panel_height = 240;
    p.invert = true;
    p.readable = false;
    panel.config(p);
    auto l = light.config();
    l.pin_bl = PIN_TFT_BLK;
    l.pwm_channel = 7;
    light.config(l);
    panel.setLight(&light);
    setPanel(&panel);
  }
};

static Display tft;
static LGFX_Sprite canvas(&tft);
static WebSocketsClient ws;

static const uint16_t INK = 0x528A, PAPER = 0xF7BE, ACCENT = 0xFAC3;
static const int MAX_CURVE = 64, MAX_OBS = 6;

struct Stats {
  int generation = 0, best = 0, score = 0, games = 0, jumps = 0, deaths = 0;
  float realtime = 0, meanRecent = 0;
  float curve[MAX_CURVE];
  int curveLen = 0;
  // live mini game (from the canonical dino-core state: 19 header ints + 10 per obstacle)
  long dinoY = 0;
  bool ducking = false, crashed = false;
  int nObs = 0;
  long obsX[MAX_OBS];
  int obsY[MAX_OBS], obsW[MAX_OBS], obsH[MAX_OBS];
} st;

static bool online = false, lanMode = true;
static int screen = 0;
static uint32_t ledOffAt = 0, lastFrameAt = 0, lastDraw = 0;

static void sendDopamine(const char* kind) {
  JsonDocument doc;
  doc["type"] = "dopamine.event";
  doc["v"] = 1;
  doc["ts"] = (double)millis();
  JsonObject p = doc["payload"].to<JsonObject>();
  p["kind"] = kind;
  p["magnitude"] = 1;
  p["source"] = "hardware";
  String out;
  serializeJson(doc, out);
  ws.sendTXT(out);
}

static void onFrame(JsonObject p) {
  JsonArray s = p["state"];
  if (s.size() >= 19) {
    st.score = (int)(s[3].as<double>() / 40000.0);
    st.crashed = s[4].as<int>() == 1;
    st.dinoY = s[6].as<long>();
    st.ducking = s[9].as<int>() == 1;
    st.nObs = min((int)s[18].as<int>(), MAX_OBS);
    for (int i = 0; i < st.nObs; i++) {
      int o = 19 + 10 * i;
      st.obsX[i] = s[o + 1].as<long>();
      st.obsY[i] = s[o + 2].as<int>();
      st.obsW[i] = s[o + 4].as<int>();
      st.obsH[i] = s[o + 5].as<int>();
    }
  }
  if (p["gf"].as<int>() > 0) {  // Giant Fiber spike → flash
    digitalWrite(PIN_LED, HIGH);
    ledOffAt = millis() + 60;
  }
  lastFrameAt = millis();
}

static void onStats(JsonObject p) {
  st.generation = p["generation"] | 0;
  st.best = p["bestScore"] | 0;
  st.games = p["gamesPlayed"] | 0;
  st.jumps = p["totalJumps"] | 0;
  st.deaths = p["totalDeaths"] | 0;
  st.realtime = p["realtimeFactor"] | 0.0f;
  st.meanRecent = p["meanScoreRecent"] | 0.0f;
  JsonArray c = p["learningCurve"];
  st.curveLen = min((int)c.size(), MAX_CURVE);
  for (int i = 0; i < st.curveLen; i++) st.curve[i] = c[i]["heldoutMean"] | 0.0f;
}

static void onWs(WStype_t type, uint8_t* payload, size_t length) {
  if (type == WStype_CONNECTED) online = true;
  if (type == WStype_DISCONNECTED) online = false;
  if (type != WStype_TEXT) return;
  JsonDocument doc;
  if (deserializeJson(doc, payload, length)) return;
  const char* t = doc["type"] | "";
  JsonObject p = doc["payload"];
  if (!strcmp(t, "fly.frame")) onFrame(p);
  else if (!strcmp(t, "fly.stats")) onStats(p);
  else if (!strcmp(t, "fly.status")) online = p["online"] | false;
}

static void connectWs() {
  lanMode = digitalRead(PIN_SW_MODE) == LOW;
  if (lanMode) ws.begin(LAN_HOST, LAN_PORT, "/");
  else ws.beginSSL(CLOUD_HOST, CLOUD_PORT, CLOUD_PATH);
  ws.onEvent(onWs);
  ws.setReconnectInterval(3000);
}

static void header(const char* title) {
  canvas.fillScreen(PAPER);
  canvas.setTextColor(INK);
  canvas.setFont(&fonts::Font2);
  canvas.setCursor(6, 4);
  canvas.print(title);
  canvas.fillRect(222, 6, 10, 10, online && millis() - lastFrameAt < 3000 ? ACCENT : INK);
  canvas.drawFastHLine(0, 22, 240, INK);
}

static void bigNumber(int y, const char* label, int value, uint16_t color) {
  canvas.setFont(&fonts::Font2);
  canvas.setTextColor(INK);
  canvas.setCursor(8, y);
  canvas.print(label);
  canvas.setFont(&fonts::Font7);
  canvas.setTextColor(color);
  canvas.setCursor(8, y + 16);
  canvas.print(value);
}

static void draw() {
  switch (screen) {
    case 0:
      header(lanMode ? "dino-fly  LAN" : "dino-fly  CLOUD");
      bigNumber(28, "generation", st.generation, INK);
      bigNumber(98, "best score", st.best, INK);
      bigNumber(168, "now", st.score, ACCENT);
      break;
    case 1:
      header("this session");
      canvas.setFont(&fonts::Font4);
      canvas.setTextColor(INK);
      canvas.setCursor(8, 36);
      canvas.printf("games  %d\n jumps  %d\n deaths %d\n mean50 %.0f\n speed  %.2fx", st.games, st.jumps, st.deaths, st.meanRecent, st.realtime);
      break;
    case 2: {
      header("learning curve");
      if (st.curveLen < 2) {
        canvas.setCursor(8, 110);
        canvas.print("generation 0 only");
        break;
      }
      float lo = st.curve[0], hi = st.curve[0];
      for (int i = 1; i < st.curveLen; i++) {
        lo = min(lo, st.curve[i]);
        hi = max(hi, st.curve[i]);
      }
      if (hi - lo < 1) hi = lo + 1;
      for (int i = 1; i < st.curveLen; i++) {
        int x0 = 8 + (i - 1) * 224 / (st.curveLen - 1), x1 = 8 + i * 224 / (st.curveLen - 1);
        int y0 = 220 - (int)((st.curve[i - 1] - lo) / (hi - lo) * 170), y1 = 220 - (int)((st.curve[i] - lo) / (hi - lo) * 170);
        canvas.drawLine(x0, y0, x1, y1, ACCENT);
      }
      canvas.setCursor(8, 28);
      canvas.printf("held-out %.0f .. %.0f", lo, hi);
      break;
    }
    default: {  // live mini game: world 600x150 scaled by 0.4 → 240x60, centred
      header("the fly, live");
      const float k = 0.4f;
      const int ground = 150;
      canvas.drawFastHLine(0, ground, 240, INK);
      int dh = st.ducking ? 25 : 47, dw = st.ducking ? 59 : 44;
      int dy = ground - (int)((st.dinoY / 1000.0f + dh) * k);
      canvas.fillRect((int)(50 * k), dy, (int)(dw * k), (int)(dh * k), st.crashed ? INK : ACCENT);
      for (int i = 0; i < st.nObs; i++) {
        int x = (int)(st.obsX[i] / 1000.0f * k), y = ground - (int)((st.obsY[i] + st.obsH[i]) * k);
        canvas.fillRect(x, y, max(2, (int)(st.obsW[i] * k)), (int)(st.obsH[i] * k), INK);
      }
      canvas.setFont(&fonts::Font4);
      canvas.setCursor(8, 180);
      canvas.printf("score %d", st.score);
    }
  }
  canvas.pushSprite(0, 0);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_BTN_REWARD, INPUT_PULLUP);
  pinMode(PIN_BTN_PUNISH, INPUT_PULLUP);
  pinMode(PIN_SW_MODE, INPUT_PULLUP);
  tft.init();
  tft.setBrightness(180);
  canvas.setColorDepth(16);
  canvas.createSprite(240, 240);
  header("dino-fly");
  canvas.setCursor(8, 110);
  canvas.print("joining Wi-Fi ...");
  canvas.pushSprite(0, 0);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) delay(250);
  connectWs();
}

void loop() {
  ws.loop();
  uint32_t now = millis();
  if (ledOffAt && now > ledOffAt) {
    digitalWrite(PIN_LED, LOW);
    ledOffAt = 0;
  }
  // buttons: short press = dopamine; long press on PUNISH = next screen
  static uint32_t rewardDown = 0, punishDown = 0;
  bool r = digitalRead(PIN_BTN_REWARD) == LOW, p = digitalRead(PIN_BTN_PUNISH) == LOW;
  if (r && !rewardDown) rewardDown = now;
  if (!r && rewardDown) {
    if (now - rewardDown > 30) sendDopamine("reward");
    rewardDown = 0;
  }
  if (p && !punishDown) punishDown = now;
  if (!p && punishDown) {
    if (now - punishDown > 700) screen = (screen + 1) % 4;
    else if (now - punishDown > 30) sendDopamine("punish");
    punishDown = 0;
  }
  if ((digitalRead(PIN_SW_MODE) == LOW) != lanMode) {  // mode switch flipped → reconnect to the other endpoint
    ws.disconnect();
    connectWs();
  }
  if (now - lastDraw > (screen == 3 ? 66 : 250)) {
    lastDraw = now;
    draw();
  }
}
