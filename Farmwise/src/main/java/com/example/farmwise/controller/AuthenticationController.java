package com.example.farmwise.controller;

import com.example.farmwise.dto.AuthenticationResponseDto;
import com.example.farmwise.dto.SigninDto;
import com.example.farmwise.entity.User;
import com.example.farmwise.jwt.JwtProvider;
import com.example.farmwise.service.UserService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.security.Principal;
import java.util.List;

@Slf4j
@RequiredArgsConstructor
@RequestMapping("/api/v1/auth")
@RestController
@CrossOrigin(value = "http://localhost:4200")
public class AuthenticationController {

    private final UserService userService;
    private final AuthenticationManager authenticationManager;
    private final JwtProvider jwtProvider;

    @PostMapping("/signup")
    public ResponseEntity<String> signup(@RequestBody User user) {
        userService.saveUser(user);
        return new ResponseEntity<>("User successfully created", HttpStatus.CREATED);
    }

    @PostMapping("/signin")
    public ResponseEntity<AuthenticationResponseDto> signin(@RequestBody SigninDto signinDto) {
        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(signinDto.getUsername(), signinDto.getPassword())
        );

        SecurityContextHolder.getContext().setAuthentication(authentication);
        log.info("Authentication : {}", authentication);

        // Extract authenticated user principal (from Spring Security)
        org.springframework.security.core.userdetails.User principal =
                (org.springframework.security.core.userdetails.User) authentication.getPrincipal();

        // Extract roles from GrantedAuthority
        List<String> roles = principal.getAuthorities().stream()
                .map(authority -> authority.getAuthority())
                .toList(); // or use .collect(Collectors.toList())

        // Generate JWT token
        String token = jwtProvider.generateToken(authentication);

        // Return full response
        return new ResponseEntity<>(
                new AuthenticationResponseDto(token, principal.getUsername(), roles),
                HttpStatus.OK
        );
    }


    @GetMapping("/user")
    public ResponseEntity<User> getUserDetails(Principal principal){
        String username = principal.getName();
        return new ResponseEntity<>(userService.getUserDetails(username), HttpStatus.OK);
    }



}

