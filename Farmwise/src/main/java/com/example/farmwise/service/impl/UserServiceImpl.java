
package com.example.farmwise.service.impl;

import com.example.farmwise.entity.Role;
import com.example.farmwise.entity.User;
import com.example.farmwise.exception.AccessDeniedException;
import com.example.farmwise.exception.InvalidArgumentException;
import com.example.farmwise.exception.ResourceNotFoundException;
import com.example.farmwise.repository.RoleRepository;
import com.example.farmwise.repository.UserRepository;
import com.example.farmwise.service.UserService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

@RequiredArgsConstructor
@Service
public class UserServiceImpl implements UserService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final RoleRepository roleRepository;

    @Override
    public User saveUser(User user) {
        Boolean isUsernameExist = userRepository.existsByUsername(user.getUsername());

        if (!isUsernameExist) {
            Optional<Role> role = roleRepository.findByName("ROLE_USER");
            user.setRoles(Collections.singleton(role.orElseThrow(() -> new InvalidArgumentException("Role not found!"))));
            user.setPassword(passwordEncoder.encode(user.getPassword()));
            return userRepository.save(user);
        }
        throw new InvalidArgumentException("Username already exist!");
    }

    @Override
    public User getUserById(Long id) {
        return userRepository.findById(id).orElseThrow(
                () -> new UsernameNotFoundException("Username id " + id + " was not found!")
        );
    }

    @Override
    public List<User> getAllUser() {
        return userRepository.findAll();
    }

    @Override
    public User updateUser(User user) {
        User updatedUser = new User();
        updatedUser.setFirstName(user.getFirstName());
        updatedUser.setLastName(user.getLastName());
        updatedUser.setUsername(user.getUsername());
        updatedUser.setPassword(passwordEncoder.encode(user.getPassword()));
        return userRepository.save(updatedUser);
    }
    @Override
    public List<User> getAllUsers() {
        return userRepository.findAll();
    }

    @Override
    public String deleteUser(Long id) {
        userRepository.deleteById(id);
        return "User Successfully deleted";
    }

    @Override
    public boolean isUsernameExist(String username) {
        return userRepository.existsByUsername(username);
    }

    @Override
    public User getUserDetails(String username) {
        return userRepository.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("Username : " + username + " was not found!"));
    }

    @Override
    public User getUser() {
        String username = SecurityContextHolder.getContext().getAuthentication().getName();

        if (Objects.isNull(username)) {
            throw new AccessDeniedException("Invalid Access");
        }

        Optional<User> user = userRepository.findByUsername(username);
        if(user.isEmpty()){
            throw new ResourceNotFoundException("User not found");
        }

        return user.get();
    }
}
