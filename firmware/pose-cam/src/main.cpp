// dino-fly pose camera — Seeed XIAO ESP32S3 Sense (OV2640).
// Serves an MJPEG stream at http://<ip>:81/stream ; the laptop runs MediaPipe Pose on it (brain/pose/pose_input.py)
// and turns the player's body movement into JUMP / DUCK. The camera itself makes no decisions.
//
// NOT YET TESTED ON HARDWARE (pin map = Seeed's documented XIAO ESP32S3 Sense camera pins).

#include <Arduino.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <esp_http_server.h>

#include "secrets.h"

// XIAO ESP32S3 Sense camera pins
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 10
#define SIOD_GPIO_NUM 40
#define SIOC_GPIO_NUM 39
#define Y9_GPIO_NUM 48
#define Y8_GPIO_NUM 11
#define Y7_GPIO_NUM 12
#define Y6_GPIO_NUM 14
#define Y5_GPIO_NUM 16
#define Y4_GPIO_NUM 18
#define Y3_GPIO_NUM 17
#define Y2_GPIO_NUM 15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM 47
#define PCLK_GPIO_NUM 13

static const char* BOUNDARY = "frame";
static httpd_handle_t server = nullptr;

static esp_err_t streamHandler(httpd_req_t* req) {
  char part[64];
  httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  while (true) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) return ESP_FAIL;
    size_t n = snprintf(part, sizeof(part), "\r\n--%s\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", BOUNDARY, (unsigned)fb->len);
    esp_err_t res = httpd_resp_send_chunk(req, part, n);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    if (res != ESP_OK) return res;  // client went away
  }
}

static bool startCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;
  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_QVGA;  // 320x240 is plenty for pose landmarks and keeps latency low
  c.jpeg_quality = 12;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  return esp_camera_init(&c) == ESP_OK;
}

void setup() {
  Serial.begin(115200);
  if (!startCamera()) {
    Serial.println("camera init failed");
    return;
  }
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) delay(250);
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = 81;
  httpd_uri_t stream = {.uri = "/stream", .method = HTTP_GET, .handler = streamHandler, .user_ctx = nullptr};
  if (httpd_start(&server, &cfg) == ESP_OK) httpd_register_uri_handler(server, &stream);
  Serial.printf("MJPEG stream: http://%s:81/stream\n", WiFi.localIP().toString().c_str());
}

void loop() { delay(1000); }
