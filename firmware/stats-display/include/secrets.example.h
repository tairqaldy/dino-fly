// Copy to secrets.h (git-ignored) and fill in.
#pragma once
#define WIFI_SSID "your-wifi"
#define WIFI_PASSWORD "your-password"
// LAN mode: the laptop running `flybrain worker` (ws://<laptop-ip>:8765, start the worker with --host 0.0.0.0)
#define LAN_HOST "192.168.1.50"
#define LAN_PORT 8765
// Cloud mode: the public API hub (wss://<host>/ws)
#define CLOUD_HOST "api.example.com"
#define CLOUD_PORT 443
#define CLOUD_PATH "/ws"
