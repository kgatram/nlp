package model;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * Class defining booking attributes required by RBS and used by model.
 */

public class Booking implements Serializable {
    private Person person;
    private Building building;
    private LocalDateTime startTime;
    private LocalDateTime endTime;

    /**
     * Booking constructor
     * @param person Person object
     * @param building Building object
     * @param startTime LocalDateTime object for start time of booking.
     * @param endTime LocalDateTime object for end time of booking.
     */

    public Booking (Person person, Building building, LocalDateTime startTime, LocalDateTime endTime) {
        this.person = person;
        this.building = building;
        this.startTime = startTime;
        this.endTime = endTime;
        this.building.setTime(startTime, endTime);
    }

    /**
     * Returns person object for booking.
     * @return Person object
     */

    public Person getPerson() {
        return person;
    }

    /**
     * Returns building object for booking.
     * @return Building object
     */
    public Building getbuilding() {
        return building;
    }

    /**
     * Returns LocalDateTime as start time of booking.
     * @return LocalDateTime
     */
    public LocalDateTime getStartTime() {
        return startTime;
    }

    /**
     * Returns LocalDateTime as end time of booking.
     * @return LocalDateTime
     */
    public LocalDateTime getEndTime() {
        return endTime;
    }
}
