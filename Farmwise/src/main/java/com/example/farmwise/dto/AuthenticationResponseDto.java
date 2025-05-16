package com.example.farmwise.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AuthenticationResponseDto {

    private String token;
    private String tokenType = "Bearer";
    private String username;
    private List<String> roles;

    public AuthenticationResponseDto(String token, String username, List<String> roles) {
        this.token = token;
        this.username = username;
        this.roles = roles;
    }
}
