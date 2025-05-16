package com.example.farmwise.controller;

import com.example.farmwise.service.WeatherService;
import org.json.JSONObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/weather")
public class WeatherController {

    @Autowired
    private WeatherService weatherService;

    @GetMapping
    public Map<String, Object> getWeather(@RequestParam double lat, @RequestParam double lon) {
        JSONObject data = weatherService.getWeather(lat, lon);

        double temperature = data.getJSONObject("main").getDouble("temp");
        int humidity = data.getJSONObject("main").getInt("humidity");
        double rainfall = data.has("rain") ? data.getJSONObject("rain").optDouble("1h", 0.0) : 0.0;

        return Map.of(
                "temperature", temperature,
                "humidity", humidity,
                "rainfall", rainfall
        );
    }
}
