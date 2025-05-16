package com.example.farmwise.controller;

import com.example.farmwise.service.SoilService;
import org.json.JSONArray;
import org.json.JSONObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/soil")
public class SoilController {

    @Autowired
    private SoilService soilService;

    @GetMapping
    public Map<String, Object> getSoil(@RequestParam double lat, @RequestParam double lon) {
        JSONObject data = soilService.getSoilData(lat, lon);
        JSONObject properties = data.optJSONObject("properties");

        double ph = -1;
        double nitrogen = -1;

        // Mock values (Tunisia-based estimates)
        double potassium = 50.0;   // mg/kg
        double phosphorus = 35.0;  // mg/kg

        if (properties != null) {
            JSONArray layers = properties.optJSONArray("layers");
            if (layers != null) {
                for (int i = 0; i < layers.length(); i++) {
                    JSONObject layer = layers.getJSONObject(i);
                    String code = layer.optString("code");

                    JSONArray depths = layer.optJSONArray("depths");
                    if (depths != null && depths.length() > 0) {
                        JSONObject depth0 = depths.getJSONObject(0);
                        JSONObject values = depth0.optJSONObject("values");

                        if (values != null && values.has("mean") && !values.isNull("mean")) {
                            double raw = values.getDouble("mean");

                            if ("phh2o".equals(code)) {
                                ph = raw / 10.0;  // convert from pH*10 to actual pH
                            } else if ("nitrogen".equals(code)) {
                                nitrogen = raw;
                            }
                        }
                    }
                }
            }
        }

        return Map.of(
                "ph", ph,
                "nitrogen", nitrogen,
                "potassium", potassium,
                "phosphorus", phosphorus
        );
    }

}
