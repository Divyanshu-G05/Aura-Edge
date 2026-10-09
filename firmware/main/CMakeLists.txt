file(GLOB_RECURSE FE_SRCS "microfrontend/*.c" "microfrontend/*.cc")

idf_component_register(
    SRCS "main.cpp" ${FE_SRCS}
    INCLUDE_DIRS "." "microfrontend" "kissfft" "kissfft/tools"
    EMBED_FILES "model.tflite"
    REQUIRES esp_wifi esp_event esp_netif nvs_flash lwip esp_driver_i2s driver esp_timer
)
