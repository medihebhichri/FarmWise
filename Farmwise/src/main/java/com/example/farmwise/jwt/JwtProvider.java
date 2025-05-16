
package com.example.farmwise.jwt;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import org.springframework.security.authentication.AuthenticationCredentialsNotFoundException;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Component;

import java.util.Date;

@Component
public class JwtProvider {

    public String generateToken(Authentication authentication){
        String username = authentication.getName();
        Date present = new Date();
        return Jwts.builder()
                .setSubject(username)
                .setIssuedAt(new Date())
                .setExpiration(new Date(present.getTime() + JwtUtils.EXPIRATION_TIME))
                .signWith(SignatureAlgorithm.HS512, JwtUtils.JWT_SECRET)
                .compact();
    }

    public String getUsernameFromJwt(String token){
        Claims claims = Jwts.parser()
                .setSigningKey(JwtUtils.JWT_SECRET)
                .parseClaimsJws(token)
                .getBody();
        return claims.getSubject();
    }

    public Boolean isTokenValid(String token){
        try {
            Jwts.parser()
                    .setSigningKey(JwtUtils.JWT_SECRET)
                    .parseClaimsJws(token);
            return true;
        } catch (Exception e){
            throw new AuthenticationCredentialsNotFoundException("Jwt was expire or incorrect");
        }
    }


}
