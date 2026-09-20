// Wiring (see docs/HARDWARE.md). GMT130 has no CS pin: the panel is always selected, SPI mode 3.
#pragma once
#define PIN_TFT_SCK 18   // display SCK
#define PIN_TFT_SDA 23   // display SDA (MOSI)
#define PIN_TFT_RES 4    // display RES
#define PIN_TFT_DC 2     // display DC
#define PIN_TFT_BLK 15   // display BLK (backlight, PWM)
#define PIN_LED 25       // LED + 330 ohm to GND: flashes on every Giant Fiber spike / jump
#define PIN_BTN_REWARD 32  // tactile button to GND (internal pull-up): reward (PAM)
#define PIN_BTN_PUNISH 33  // tactile button to GND (internal pull-up): punish (PPL1); long press = next screen
#define PIN_SW_MODE 27     // slide switch to GND: closed = LAN, open = cloud
