// Hey Aura wake-word detector + command streamer, ESP32-C3 + INMP441 + MAX98357.
//
// Flow: listen for wake word -> beep -> stream mic audio to a TCP server
// until the SERVER says stop (it does the end-of-speech detection) -> receive
// a reply audio stream back -> play it as it arrives -> resume listening.
//
// Wire protocol (big-endian length prefixes):
//   client -> server : repeated [u32 len][PCM16LE mono @16kHz, len bytes]
//                        terminated by one [u32 len=0]   (end of utterance)
//   server -> client : during the send phase, at most one [u32 0xFFFFFFFF]
//                        = STOP (server heard the user finish; stop streaming)
//                      then repeated [u32 len][PCM16LE mono @16kHz]
//                        terminated by one [u32 len=0]   (end of reply)
// Connection is opened fresh for each wake and closed after playback.
//
// Testing without the wake word: press BOOT (GPIO9) to trigger the same flow.

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/stream_buffer.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"

#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include "lwip/inet.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/schema/schema_generated.h"
extern "C" {
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
}

// ---------------- pins ----------------
#define PIN_BCLK    GPIO_NUM_4
#define PIN_WS      GPIO_NUM_5
#define PIN_MIC_SD  GPIO_NUM_6    // mic data in
#define PIN_AMP_DIN GPIO_NUM_7    // amp data out
#define PIN_AMP_SD  GPIO_NUM_1    // MAX98357 SD: low = shutdown, high = on (left ch)
#define PIN_LED     GPIO_NUM_8    // super mini onboard LED, usually ACTIVE LOW
#define PIN_BTN     GPIO_NUM_9    // BOOT button, active low (manual trigger)
#define LED_ON      0
#define LED_OFF     1

// ---------------- wifi / server ----------------
#define WIFI_SSID        "WIFI_SSID"
#define WIFI_PASS        "WIFI_PASSWORD"
#define SERVER_IP        "BACKEND_IP"   // laptop running the Aura backend
#define SERVER_PORT      8765

// ---------------- audio / model config ----------------
#define SAMPLE_RATE      16000
#define MIC_SHIFT        14       // keep identical to your recording path (saturating)
#define STEP_SAMPLES     160      // 10 ms
#define NUM_FEATURES     40
#define STRIDE           3        // must match --stride in training

#define CUTOFF           0.97f    // starting point from your ROC; tune on device
#define WINDOW           5        // sliding-window mean of last N outputs
#define REFRACTORY_MS    1500     // ignore new triggers for this long after a round finishes
#define ARENA_SIZE       (70 * 1024)

#define BEEP_AMP         8000     // int16 peak
#define BEEP1_HZ         880
#define BEEP1_MS         120
#define BEEP2_HZ         1320
#define BEEP2_MS         150

// ---------------- streaming config ----------------
#define SEND_CHUNK_MS      100     // audio per socket write (also how often we poll for STOP)
#define MAX_UTTER_MS       10000   // failsafe if the server never sends STOP
#define SOCK_TIMEOUT_S     10      // connect / send timeout
#define REPLY_TIMEOUT_S    30      // recv timeout during reply (STT + LLM + first TTS sentence)
#define STOP_MARKER        0xFFFFFFFFu

static const char *TAG = "kws";

extern const uint8_t model_start[] asm("_binary_model_tflite_start");

alignas(16) static uint8_t tensor_arena[ARENA_SIZE];

static i2s_chan_handle_t rx = NULL;

// ================== WiFi ==================
static EventGroupHandle_t wifi_eg;
#define WIFI_CONNECTED_BIT BIT0

static void wifi_event_handler(void *, esp_event_base_t base, int32_t id, void *data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "wifi disconnected, retrying");
        xEventGroupClearBits(wifi_eg, WIFI_CONNECTED_BIT);
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "got ip: " IPSTR, IP2STR(&e->ip_info.ip));
        xEventGroupSetBits(wifi_eg, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init_sta(void) {
    ESP_ERROR_CHECK(nvs_flash_init());
    wifi_eg = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));

    wifi_config_t wc = {};
    strncpy((char *)wc.sta.ssid, WIFI_SSID, sizeof(wc.sta.ssid) - 1);
    strncpy((char *)wc.sta.password, WIFI_PASS, sizeof(wc.sta.password) - 1);
    wc.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_ps(WIFI_PS_NONE);   // modem sleep adds latency spikes to streaming

    ESP_LOGI(TAG, "connecting to wifi SSID '%s' ...", WIFI_SSID);
    EventBits_t bits = xEventGroupWaitBits(wifi_eg, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, pdMS_TO_TICKS(20000));
    if (!(bits & WIFI_CONNECTED_BIT)) {
        ESP_LOGE(TAG, "wifi connect timed out; check WIFI_SSID/WIFI_PASS. Continuing anyway.");
    }
}

// ================== I2S: mic (RX) ==================
static void mic_start(void) {
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_desc_num = 8;      // ~120 ms of slack so a slow socket write can't overflow the mic DMA
    cc.dma_frame_num = 240;
    ESP_ERROR_CHECK(i2s_new_channel(&cc, NULL, &rx));
    i2s_std_config_t c = {};
    c.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE);
    c.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);
    c.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    c.gpio_cfg.mclk = I2S_GPIO_UNUSED;
    c.gpio_cfg.bclk = PIN_BCLK;
    c.gpio_cfg.ws = PIN_WS;
    c.gpio_cfg.dout = I2S_GPIO_UNUSED;
    c.gpio_cfg.din = PIN_MIC_SD;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx, &c));
    ESP_ERROR_CHECK(i2s_channel_enable(rx));
}

static void mic_stop(void) {
    if (!rx) return;
    i2s_channel_disable(rx);
    i2s_del_channel(rx);
    rx = NULL;
}

// throw away audio right after (re)starting the mic: it needs to settle
static void mic_flush(int chunks) {
    static int32_t tmp[STEP_SAMPLES];
    size_t got;
    for (int i = 0; i < chunks; i++)
        i2s_channel_read(rx, tmp, sizeof(tmp), &got, portMAX_DELAY);
}

// read one 10ms frame from the mic into pcm[STEP_SAMPLES], with saturation.
// returns the frame's mean abs amplitude (handy for debugging levels).
static int32_t mic_read_frame(int16_t *pcm) {
    static int32_t raw[STEP_SAMPLES];
    size_t got = 0;
    i2s_channel_read(rx, raw, sizeof(raw), &got, portMAX_DELAY);
    if (got != sizeof(raw)) { memset(pcm, 0, STEP_SAMPLES * sizeof(int16_t)); return 0; }
    int64_t sum = 0;
    for (int i = 0; i < STEP_SAMPLES; i++) {
        int32_t v = raw[i] >> MIC_SHIFT;
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        pcm[i] = (int16_t)v;
        sum += (v < 0) ? -v : v;
    }
    return (int32_t)(sum / STEP_SAMPLES);
}

// ================== I2S: speaker (TX) ==================
// The C3 has no FPU, so sinf() per sample is slow. Use a 256-entry table for the beep.
static int16_t sine_tab[256];

static void sine_init(void) {
    for (int i = 0; i < 256; i++)
        sine_tab[i] = (int16_t)(BEEP_AMP * sinf(2.0f * (float)M_PI * i / 256.0f));
}

static void play_tone(i2s_chan_handle_t tx, int freq, int ms) {
    static int16_t buf[256];
    const int total = SAMPLE_RATE * ms / 1000;
    const int fade = SAMPLE_RATE / 200;                       // 5 ms fade in/out
    const uint32_t inc = (uint32_t)(((uint64_t)freq << 32) / SAMPLE_RATE);
    uint32_t phase = 0;
    int done = 0;
    while (done < total) {
        int n = total - done;
        if (n > 256) n = 256;
        for (int i = 0; i < n; i++) {
            int idx = done + i;
            int32_t s = sine_tab[phase >> 24];
            if (idx < fade) s = s * idx / fade;
            else if (total - idx < fade) s = s * (total - idx) / fade;
            buf[i] = (int16_t)s;
            phase += inc;
        }
        size_t w = 0;
        i2s_channel_write(tx, buf, n * sizeof(int16_t), &w, portMAX_DELAY);
        done += n;
    }
}

// opens a 16-bit mono TX channel on the shared clock pins and enables the amp.
// caller must own the bus (mic_stop() already called).
static i2s_chan_handle_t tx_open(void) {
    i2s_chan_handle_t tx;
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_desc_num = 6;
    cc.dma_frame_num = 240;
    cc.auto_clear = true;     // underrun (gap between reply sentences) = silence, not a repeating buzz
    ESP_ERROR_CHECK(i2s_new_channel(&cc, &tx, NULL));
    i2s_std_config_t c = {};
    c.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE);
    c.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
    c.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    c.gpio_cfg.mclk = I2S_GPIO_UNUSED;
    c.gpio_cfg.bclk = PIN_BCLK;
    c.gpio_cfg.ws = PIN_WS;
    c.gpio_cfg.dout = PIN_AMP_DIN;
    c.gpio_cfg.din = I2S_GPIO_UNUSED;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(tx, &c));
    ESP_ERROR_CHECK(i2s_channel_enable(tx));

    static int16_t zeros[256];
    size_t w = 0;
    for (int i = 0; i < 3; i++) i2s_channel_write(tx, zeros, sizeof(zeros), &w, portMAX_DELAY);
    gpio_set_level(PIN_AMP_SD, 1);
    vTaskDelay(pdMS_TO_TICKS(20));
    return tx;
}

static void tx_close(i2s_chan_handle_t tx) {
    static int16_t zeros[256];
    size_t w = 0;
    for (int i = 0; i < 6; i++) i2s_channel_write(tx, zeros, sizeof(zeros), &w, portMAX_DELAY);
    vTaskDelay(pdMS_TO_TICKS(40));
    gpio_set_level(PIN_AMP_SD, 0);
    i2s_channel_disable(tx);
    i2s_del_channel(tx);
}

// Plays the beep. Leaves the I2S bus FREE (no channel open) when it returns;
// the caller must call mic_start() afterwards.
static void beep(void) {
    ESP_LOGI(TAG, "beep: start");
    mic_stop();
    i2s_chan_handle_t tx = tx_open();
    play_tone(tx, BEEP1_HZ, BEEP1_MS);
    play_tone(tx, BEEP2_HZ, BEEP2_MS);
    tx_close(tx);
    ESP_LOGI(TAG, "beep: done");
}

// ================== socket helpers ==================
// send all bytes or return false
static bool send_all(int sock, const uint8_t *buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        int n = send(sock, buf + sent, len - sent, 0);
        if (n <= 0) return false;
        sent += n;
    }
    return true;
}

static bool recv_all(int sock, uint8_t *buf, size_t len) {
    size_t got = 0;
    while (got < len) {
        int n = recv(sock, buf + got, len - got, 0);
        if (n <= 0) return false;   // 0 = closed, <0 = error/timeout
        got += n;
    }
    return true;
}

static bool send_frame(int sock, const uint8_t *data, uint32_t len) {
    uint32_t be = htonl(len);
    if (!send_all(sock, (uint8_t *)&be, 4)) return false;
    if (len == 0) return true;
    return send_all(sock, data, len);
}

// returns frame length via *len (0 = end marker). false on connection error.
static bool recv_frame_len(int sock, uint32_t *len) {
    uint32_t be;
    if (!recv_all(sock, (uint8_t *)&be, 4)) return false;
    *len = ntohl(be);
    return true;
}

// ================== reply playback: jitter buffer + player task ==================
// The socket reader pushes reply audio into a ring buffer as fast as the network
// delivers it; a separate task drains it into I2S. Wi-Fi / server hiccups no longer
// starve the speaker (that was the tearing/stutter).
#define PLAY_BUF_BYTES   (24 * 1024)   // ~750 ms of 16 kHz mono int16
#define PREFILL_BYTES    6400          // 200 ms buffered before playback starts

static uint8_t play_storage[PLAY_BUF_BYTES + 1];
static StaticStreamBuffer_t play_sb_struct;
static StreamBufferHandle_t play_sb = NULL;
static volatile bool play_eof = false;
static volatile int play_underruns = 0;
static i2s_chan_handle_t play_tx = NULL;
static TaskHandle_t play_waiter = NULL;

static void player_task(void *) {
    static uint8_t pbuf[2052];
    size_t carry = 0;                  // keep sample alignment if a read ends on an odd byte
    while (!play_eof && xStreamBufferBytesAvailable(play_sb) < PREFILL_BYTES)
        vTaskDelay(pdMS_TO_TICKS(10));
    bool in_gap = false;
    while (true) {
        size_t n = xStreamBufferReceive(play_sb, pbuf + carry, 2048, pdMS_TO_TICKS(50));
        size_t total = carry + n;
        size_t wr = total & ~(size_t)1;
        if (wr) {
            size_t w = 0;
            i2s_channel_write(play_tx, pbuf, wr, &w, portMAX_DELAY);
            in_gap = false;
        }
        carry = total - wr;
        if (carry) pbuf[0] = pbuf[wr];
        if (n == 0) {
            if (play_eof) break;
            if (!in_gap) { in_gap = true; play_underruns++; }
        }
    }
    xTaskNotifyGive(play_waiter);
    vTaskDelete(NULL);
}

// ================== command streaming (wake -> stream -> reply) ==================
// Called after the beep, with the mic already running again. Streams audio
// to the server until the server sends STOP, then plays back whatever the
// server sends in reply. Leaves the mic running when it returns.
static void stream_command(void) {
    struct sockaddr_in dest = {};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(SERVER_PORT);
    if (inet_pton(AF_INET, SERVER_IP, &dest.sin_addr) != 1) {
        ESP_LOGE(TAG, "bad SERVER_IP");
        return;
    }

    int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock < 0) { ESP_LOGE(TAG, "socket() failed"); return; }

    struct timeval tv = { .tv_sec = SOCK_TIMEOUT_S, .tv_usec = 0 };
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    int nodelay = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

    ESP_LOGI(TAG, "connecting to %s:%d ...", SERVER_IP, SERVER_PORT);
    if (connect(sock, (struct sockaddr *)&dest, sizeof(dest)) != 0) {
        ESP_LOGE(TAG, "connect() failed (is the backend running / IP right?)");
        close(sock);
        return;
    }
    ESP_LOGI(TAG, "connected, streaming...");

    // ---- send phase: mic is already running (rx); stream until server says STOP ----
    static int16_t pcm[STEP_SAMPLES];
    static uint8_t chunk_buf[SEND_CHUNK_MS * SAMPLE_RATE / 1000 * sizeof(int16_t)];
    const int frames_per_chunk = (SEND_CHUNK_MS * SAMPLE_RATE / 1000) / STEP_SAMPLES;
    int chunk_frame = 0;
    int chunks_sent = 0;
    int32_t level = 0;

    bool ok = true;
    bool server_stop = false;
    const int64_t utter_start_us = esp_timer_get_time();

    while (ok && !server_stop) {
        level = mic_read_frame(pcm);
        memcpy(chunk_buf + chunk_frame * STEP_SAMPLES * sizeof(int16_t), pcm, STEP_SAMPLES * sizeof(int16_t));
        chunk_frame++;

        if (chunk_frame >= frames_per_chunk) {
            ok = send_frame(sock, chunk_buf, chunk_frame * STEP_SAMPLES * sizeof(int16_t));
            chunk_frame = 0;
            if (!ok) { ESP_LOGE(TAG, "send failed mid-utterance"); break; }
            if ((++chunks_sent % 10) == 0) ESP_LOGI(TAG, "streaming... level=%d", (int)level);

            // non-blocking poll: has the server decided the user is done?
            uint32_t sig = 0;
            int n = recv(sock, &sig, sizeof(sig), MSG_DONTWAIT);
            if (n == (int)sizeof(sig) && ntohl(sig) == STOP_MARKER) {
                server_stop = true;
            } else if (n == 0) {
                ESP_LOGW(TAG, "server closed connection while streaming");
                ok = false;
            } else if (n < 0 && errno != EWOULDBLOCK && errno != EAGAIN) {
                ESP_LOGW(TAG, "poll recv error errno=%d", errno);
                ok = false;
            }
        }

        if (ok && !server_stop &&
            (esp_timer_get_time() - utter_start_us) / 1000 >= MAX_UTTER_MS) {
            ESP_LOGW(TAG, "no STOP from server, hit MAX_UTTER_MS failsafe");
            if (chunk_frame > 0) {
                send_frame(sock, chunk_buf, chunk_frame * STEP_SAMPLES * sizeof(int16_t));
                chunk_frame = 0;
            }
            break;
        }
    }

    if (!ok) {
        close(sock);
        mic_flush(5);
        return;                      // mic still running; main loop resets state
    }

    ESP_LOGI(TAG, "utterance done (%s), %lld ms",
             server_stop ? "server STOP" : "failsafe",
             (long long)((esp_timer_get_time() - utter_start_us) / 1000));
    send_frame(sock, NULL, 0);       // end-of-utterance marker

    // ---- receive phase: wait for the reply, open the speaker only once audio arrives ----
    mic_stop();
    struct timeval rtv = { .tv_sec = REPLY_TIMEOUT_S, .tv_usec = 0 };
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &rtv, sizeof(rtv));
    ESP_LOGI(TAG, "waiting for reply (timeout %ds)...", REPLY_TIMEOUT_S);

    i2s_chan_handle_t tx = NULL;
    static uint8_t rxbuf[4096];
    int reply_frames = 0;
    size_t reply_bytes = 0;
    int stall = 0;
    while (true) {
        uint32_t len;
        if (!recv_frame_len(sock, &len)) { ESP_LOGW(TAG, "reply recv failed/closed"); break; }
        if (len == 0) { ESP_LOGI(TAG, "reply: end marker, %d frames, %u bytes total", reply_frames, (unsigned)reply_bytes); break; }
        if (reply_frames == 0) {
            ESP_LOGI(TAG, "reply: first frame arrived (%u bytes)", (unsigned)len);
            tx = tx_open();
            if (!play_sb) play_sb = xStreamBufferCreateStatic(PLAY_BUF_BYTES, 1, play_storage, &play_sb_struct);
            xStreamBufferReset(play_sb);
            play_tx = tx;
            play_eof = false;
            play_underruns = 0;
            play_waiter = xTaskGetCurrentTaskHandle();
            xTaskCreate(player_task, "play", 4096, NULL, 6, NULL);
        }
        reply_frames++;
        reply_bytes += len;
        uint32_t remaining = len;
        while (remaining > 0) {
            uint32_t n = remaining < sizeof(rxbuf) ? remaining : sizeof(rxbuf);
            if (!recv_all(sock, rxbuf, n)) { ESP_LOGW(TAG, "reply body recv failed"); goto reply_done; }
            size_t off = 0;
            while (off < n) {
                size_t s = xStreamBufferSend(play_sb, rxbuf + off, n - off, pdMS_TO_TICKS(1000));
                off += s;
                if (s == 0 && ++stall > 20) { ESP_LOGE(TAG, "player stalled"); goto reply_done; }
                if (s > 0) stall = 0;
            }
            remaining -= n;
        }
    }
reply_done:
    if (tx) {
        play_eof = true;
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(5000));   // let the player drain the buffer
        ESP_LOGI(TAG, "playback done, buffer underruns=%d", (int)play_underruns);
        tx_close(tx);
    }
    close(sock);
    ESP_LOGI(TAG, "stream_command: done");

    mic_start();
    mic_flush(20);
}

// ================== main task ==================
static void kws_task(void *) {
    gpio_set_level(PIN_AMP_SD, 0);
    sine_init();

    gpio_reset_pin(PIN_LED);
    gpio_set_direction(PIN_LED, GPIO_MODE_OUTPUT);
    gpio_set_level(PIN_LED, LED_OFF);
    gpio_reset_pin(PIN_BTN);
    gpio_set_direction(PIN_BTN, GPIO_MODE_INPUT);
    gpio_set_pull_mode(PIN_BTN, GPIO_PULLUP_ONLY);

    // ---- boot self-test: speaker + LED ----
    gpio_set_level(PIN_LED, LED_ON);
    beep();
    gpio_set_level(PIN_LED, LED_OFF);

    // ---- microfrontend (same params as ESPHome micro_wake_word) ----
    FrontendConfig fc;
    FrontendFillConfigWithDefaults(&fc);
    fc.window.size_ms = 30;
    fc.window.step_size_ms = 10;
    fc.filterbank.num_channels = NUM_FEATURES;
    fc.filterbank.lower_band_limit = 125.0f;
    fc.filterbank.upper_band_limit = 7500.0f;
    fc.noise_reduction.smoothing_bits = 10;
    fc.noise_reduction.even_smoothing = 0.025f;
    fc.noise_reduction.odd_smoothing = 0.06f;
    fc.noise_reduction.min_signal_remaining = 0.05f;
    fc.pcan_gain_control.enable_pcan = 1;
    fc.pcan_gain_control.strength = 0.95f;
    fc.pcan_gain_control.offset = 80.0f;
    fc.pcan_gain_control.gain_bits = 21;
    fc.log_scale.enable_log = 1;
    fc.log_scale.scale_shift = 6;
    FrontendState fs;
    if (!FrontendPopulateState(&fc, &fs, SAMPLE_RATE)) {
        ESP_LOGE(TAG, "frontend init failed");
        vTaskDelete(NULL);
    }

    // ---- model ----
    const tflite::Model *model = tflite::GetModel(model_start);
    static tflite::MicroMutableOpResolver<13> resolver;
    resolver.AddCallOnce();
    resolver.AddVarHandle();
    resolver.AddReadVariable();
    resolver.AddAssignVariable();
    resolver.AddConcatenation();
    resolver.AddStridedSlice();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddSplitV();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddLogistic();
    resolver.AddQuantize();

    tflite::MicroAllocator *alloc = tflite::MicroAllocator::Create(tensor_arena, ARENA_SIZE);
    tflite::MicroResourceVariables *rv = tflite::MicroResourceVariables::Create(alloc, 20);
    static tflite::MicroInterpreter interp(model, resolver, alloc, rv);
    if (interp.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed (raise ARENA_SIZE or check ops)");
        vTaskDelete(NULL);
    }
    TfLiteTensor *in = interp.input(0);
    TfLiteTensor *out = interp.output(0);
    ESP_LOGI(TAG, "arena used: %u bytes; input bytes=%u (expect %d) type=%d; output type=%d",
             (unsigned)interp.arena_used_bytes(), (unsigned)in->bytes,
             STRIDE * NUM_FEATURES, in->type, out->type);
    if (in->bytes != STRIDE * NUM_FEATURES) {
        ESP_LOGE(TAG, "input size mismatch, check STRIDE/NUM_FEATURES");
        vTaskDelete(NULL);
    }

    // ---- wifi (after the boot beep, so you always get audible confirmation
    //      the board is alive even if wifi creds are wrong) ----
    wifi_init_sta();

    mic_start();
    mic_flush(20);

    static int16_t pcm[STEP_SAMPLES];
    int slices = 0;
    uint8_t hist[WINDOW] = {0};
    int hpos = 0;
    int64_t last_fire = -REFRACTORY_MS * 1000LL;
    int64_t worst_us = 0;
    int64_t last_stats = 0;
    const int thresh = (int)(CUTOFF * 255.0f);

    while (true) {
        mic_read_frame(pcm);

        size_t read = 0;
        FrontendOutput fo = FrontendProcessSamples(&fs, pcm, STEP_SAMPLES, &read);
        if (fo.size == 0) continue;   // frontend still filling its first window

        int8_t *dst = in->data.int8 + slices * NUM_FEATURES;
        for (int i = 0; i < NUM_FEATURES; i++) {
            int32_t val = ((int32_t)fo.values[i] * 256 + 333) / 666;   // same scaling as ESPHome
            val -= 128;
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            dst[i] = (int8_t)val;
        }
        if (++slices < STRIDE) continue;
        slices = 0;

        int64_t t0 = esp_timer_get_time();
        if (interp.Invoke() != kTfLiteOk) { ESP_LOGE(TAG, "Invoke failed"); continue; }
        int64_t dt = esp_timer_get_time() - t0;
        if (dt > worst_us) worst_us = dt;

        uint8_t p = (out->type == kTfLiteUInt8) ? out->data.uint8[0]
                                                : (uint8_t)(out->data.int8[0] + 128);
        hist[hpos] = p;
        hpos = (hpos + 1) % WINDOW;
        int sum = 0;
        for (int i = 0; i < WINDOW; i++) sum += hist[i];
        int mean = sum / WINDOW;

        int64_t now = esp_timer_get_time();
        bool cooled = (now - last_fire) > REFRACTORY_MS * 1000LL;
        bool by_word = (mean > thresh) && cooled;
        bool by_button = (gpio_get_level(PIN_BTN) == 0) && cooled;

        if (by_word || by_button) {
            static uint32_t wake_count = 0;
            if (by_word) wake_count++;
            ESP_LOGW(TAG, ">>> %s #%u  mean=%d/255  t=%lld ms",
                     by_word ? "WAKE" : "BUTTON", (unsigned)wake_count, mean,
                     (long long)(now / 1000));
            gpio_set_level(PIN_LED, LED_ON);

            beep();                    // frees the I2S bus when done
            mic_start();               // hand the bus back to the mic for streaming
            mic_flush(10);

            stream_command();          // streams until server STOP, plays the reply, restarts mic

            gpio_set_level(PIN_LED, LED_OFF);

            // fresh start so leftover audio/model state can't cause a spurious retrigger
            FrontendReset(&fs);
            rv->ResetAll();
            memset(hist, 0, sizeof(hist));
            hpos = 0;
            slices = 0;
            last_fire = esp_timer_get_time();
        }

        now = esp_timer_get_time();
        (void)p; (void)worst_us;

        // ---- CPU / RAM stats, once every 5 s ----
        if (now - last_stats > 5000000) {
            last_stats = now;
            size_t free_heap = esp_get_free_heap_size();
            size_t min_free_heap = esp_get_minimum_free_heap_size();
            UBaseType_t stack_left = uxTaskGetStackHighWaterMark(NULL);
            ESP_LOGI(TAG, "[STATS] heap free=%u B  min-ever=%u B  task-stack-free=%u B",
                     (unsigned)free_heap, (unsigned)min_free_heap,
                     (unsigned)(stack_left * sizeof(StackType_t)));
#if configGENERATE_RUN_TIME_STATS
            static char rt_buf[512];
            vTaskGetRunTimeStats(rt_buf);
            ESP_LOGI(TAG, "[STATS] per-task CPU time (ticks) / %%:\n%s", rt_buf);
#endif
        }
    }
}

extern "C" void app_main(void) {
    gpio_reset_pin(PIN_AMP_SD);
    gpio_set_direction(PIN_AMP_SD, GPIO_MODE_OUTPUT);
    gpio_set_level(PIN_AMP_SD, 0);
    xTaskCreate(kws_task, "kws", 20 * 1024, NULL, 5, NULL);
}
