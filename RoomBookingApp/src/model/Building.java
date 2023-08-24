package model;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.ArrayList;

/**
 * Class defining building attributes required by RBS and used by model.
 */
public class Building implements Serializable {
    private String name;
    private String address;
    private String room;

    private ArrayList<LocalDateTime> startTime = new ArrayList<>();
    private ArrayList<LocalDateTime> endTime = new ArrayList<>();

    /**
     * Constructor for Building class.
     * @param name String value containing name of building.
     * @param address String value for address.
     * @param room String value for room.
     */
    public Building(String name, String address, String room) {
        this.name = name;
        this.address = address;
        this.room = room;
    }

    /**
     * Get name of building
     * @return String
     */
    public String getName() {
        return name;
    }

    /**
     * Get room of building.
     * @return String
     */

    public String getRoom() {
        return room;
    }

    /**
     * Set the value of room.
     * @param room String
     */

    public void setRoom(String room) {
        this.room = room;
    }

    /**
     * Set start time and end time for room allocation
     * @param startTime LocalDateTime
     * @param endTime LocalDateTime
     */
    public void setTime(LocalDateTime startTime, LocalDateTime endTime ) {
        this.startTime.add(startTime);
        this.endTime.add(endTime);
    }

    /**
     * Get schedule of room with start time.
     * @return ArrayList
     */
    public ArrayList<LocalDateTime> getStartTime() {
        return startTime;
    }

    /**
     * Get schedule of room with end time.
     * @return ArrayList
     */
    public ArrayList<LocalDateTime> getEndTime() {
        return endTime;
    }

    /**
     * free allocated schedule of room.
     */
    public void freeAllocation(LocalDateTime starttime, LocalDateTime endtime) {
        startTime.remove(starttime);
        endTime.remove(endtime);
    }
}
