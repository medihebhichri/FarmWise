package com.example.farmwise.service;

import org.json.JSONObject;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

@Service
public class SoilService {

    public JSONObject getSoilData(double lat, double lon) {
        String url = UriComponentsBuilder
                .fromHttpUrl("https://api.openepi.io/soil/property")
                .queryParam("lat", lat)
                .queryParam("lon", lon)
                .queryParam("depths", "0-5cm")
                .queryParam("properties", "phh2o")
                .queryParam("properties", "bdod")
                .queryParam("values", "mean")
                .toUriString();

        RestTemplate restTemplate = new RestTemplate();
        String response = restTemplate.getForObject(url, String.class);

        return new JSONObject(response);
    }
}
