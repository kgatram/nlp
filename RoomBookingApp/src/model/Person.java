package model;

import java.io.Serializable;

/**
 * Class defining Person attributes required by RBS and used by model.
 */

public class Person implements Serializable {
    private String name;
    private String email;

    /**
     * Constructor of Person class
     * @param name String
     * @param email String
     */
    public Person(String name, String email) {
        this.name = name;
        this.email = email;
    }

    /**
     * Return email of person.
     * @return String
     */
    public String getEmail() {
        return email;
    }

}
