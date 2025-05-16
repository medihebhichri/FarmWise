
package com.example.farmwise.service;

import com.example.farmwise.entity.User;

import java.util.List;

public interface UserService {

    User saveUser(User user);

    User getUserById(Long id);

    List<User> getAllUser();

    User updateUser(User user);

    String deleteUser(Long id);

    boolean isUsernameExist(String username);

    User getUserDetails(String username);

    User getUser();


    List<User> getAllUsers();


}

