package model;

import java.io.Serializable;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.*;

/**
 * This is model class of Room Booking system (RBS) and represents Model of Model-Delegate design pattern.
 * This class also implements listener design pattern aka observer pattern by providing an interface called Listener.
 * UI attached to this model implement Listener interface and add themselves as listener to get updates from model
 * Any update in model is notified to all listeners. (Reference Frog example).
 * The model holds the information of Person, Building/Room, Bookings object created in the system. It provides methods
 * Add/Remove person, room and booking in the system and performs model validation - booking should be of at least 5 min,
 * length of booking in multiple of 5, checks given time slot is available.
 * Model is serialised to save it to a file.
 *
 * @author  220025456
 * @version "%I%, %G%"
 * @since JDK17
 */

public class RBSModel implements Serializable {
    private ArrayList<Person> personList = new ArrayList<>();

    private ArrayList<Building> buildingList = new ArrayList<>();
    private ArrayList<Booking> bookingList = new ArrayList<>();

//    make it transient to not save listeners on model save.
    private transient List<Listener> listeners;
    private String message;

    private final int MIN_DURATION = 5;


    /**
     * View will observe this model implementing this interface.
     * */
    public interface Listener {
        void update();  // Called for changes in model
    }

    public RBSModel () {
        listeners = new ArrayList<Listener>();
    }

    /**
     * View will register them as listener to this Model.
     * */
    public void addListener(Listener listener) {
        listeners.add(listener);
    }

    /**
     * Model notifies all the listeners by calling their update method.
     * */
    private void changed() {
        for (Listener listener : listeners) {
            listener.update();
        }
    }

    /**
     * View calls this method to get updated message from model.
     * @return message from model.
     */
    public String getMessage() {
        return message;
    }

    /**
     * initialise listener when model is loaded.
     */
    public void initialiseListeners() {
        listeners = new ArrayList<Listener>();
    }

    /**
     * Checks and return a person object having input email.
     * UI checks for valid email before passing to model.
     * @param email valid email.
     * @return returns person object or null.
     */
    public Person checkPerson (String email) {
        for (Person person : personList) {
            if (person.getEmail().equals(email)) {
                return person;
            }
        }
        return null;
    }

    /**
     * Create a person object with email and update message if person is added to the system.
     * UI checks for valid email before passing to model.
     * @param email required valid email.
     * @param name  can be blanks.
     */
    public void addPerson (String email, String name) {
        message = "-Person already exist.";

        if (checkPerson(email) == null) {
            Person newPerson = new Person(name, email);
            personList.add(newPerson);
            message = "-Person Added.";
         }

        changed();
    }

    /**
     * Remove person from system having given email.
     * UI checks for valid email before passing to model.
     * @param email valid email is required.
     */

    public void removePerson(String email) {
        message = "-Person not found.";

        Person person = checkPerson(email);
        if (person != null) {
            removePersonBooking(email);
            personList.remove(person);
            message = "-Person Removed.";
        }

        changed();
    }

    /**
     * Remove person's booking from the system to free up allocated room.
     * UI checks for valid email before passing to model.
     * @param email valid email required
     */
    public void removePersonBooking(String email) {

//  free up room allocated to person.
        for (Booking booking: bookingList) {
            if (booking.getPerson().getEmail().equals(email)) {
                booking.getbuilding().freeAllocation(booking.getStartTime(), booking.getEndTime());
            }
        }
// remove person from booking list.
        bookingList.removeIf(booking -> booking.getPerson().getEmail().equals(email));
    }

    /**
     * Check building object with given name and room exist in system
     * UI validates building name and room is not null or blanks.
     * @param name building name
     * @param room room number
     * @return building object or null.
     */
    public Building checkBuilding (String name, String room) {
            for(Building building:buildingList) {
                if(building.getName().equals(name)
                        && building.getRoom().equals(room)) {
                    return building;
                }
        }
        return null;
    }

    /**
     * Add a building object to the model with given room, building name and address
     * UI validates building name and room is not null or blanks.
     * @param room String value
     * @param name String value
     * @param address String value, can be blanks.
     */
    public void addRoom(String room, String name, String address) {
        message = "-Room in Building already exist.";

        Building building = checkBuilding(name, room);
        if (building == null) {
            Building newRoom = new Building(name, address, room);
            buildingList.add(newRoom);
            message = "-Room Added.";
        }

        changed();
    }

    /**
     * Remove building object from model for given room and building name.
     * UI validates building name and room is not null or blanks.
     * @param room String values.
     * @param name String values.
     */
    public void removeRoom(String room, String name) {

        message = "-Room not found.";

        Building building = checkBuilding(name, room);
        if (building != null) {
                removeRoomBooking(name, room);
                buildingList.remove(building);
                message = "-Room Removed.";
        }

        changed();
    }

    /**
     * Remove booking object from model for given room and building.
     * UI validates building name and room is not null or blanks.
     * @param bldg String value
     * @param room String value
     */
    public void removeRoomBooking(String bldg, String room) {
        bookingList.removeIf(booking -> booking.getbuilding().getName().equals(bldg)
                                        && booking.getbuilding().getRoom().equals(room));
    }

    /**
     * Add building object to model with building name and address without room details.
     * UI validates building name is not null or blanks.
     * @param name String value
     * @param address String value. Can be blanks.
     */
    public void addBuilding(String name, String address) {
        boolean found = false;

        for (Building building : buildingList) {
            if (building.getName().equals(name)) {
                found = true;
                message = "-Building already exist.";
                break;
            }
        }

        if (!found) {
            Building newBldg = new Building(name, address, " ");
            buildingList.add(newBldg);
            message = "-Building Added.";
        }

        changed();
    }

    /**
     * Remove all building objects from model for given building name. Also removes all bookings for the building.
     * UI validates building name is not null or blanks.
     * @param name String value
     */
    public void removeBuilding(String name) {

        bookingList.removeIf(booking -> booking.getbuilding().getName().equals(name));

        int osize = buildingList.size();
        buildingList.removeIf(building -> building.getName().equals(name));

        message = buildingList.size() < osize ? "-Building Removed." : "-Building does not exist.";
        changed();
    }


    /**
     * Converts string value date and time into LocalDateTime object.
     * UI validates format of date and time before passing to model.
     * @param yyyymmdd String value in yyyy-mm-dd format.
     * @param time of format hh:mm
     * @return LocalDateTime
     */
    public LocalDateTime getDateTime (String yyyymmdd, String time) {
        return LocalDateTime.parse(yyyymmdd + "T" + time);
    }


    /**
     * Add a booking to RBS by creating a booking object for given details. If a person (email) is not in the system it
     * is created at the time of booking. Building and Room must exist in system at the time of booking.
     * UI validated email, date and time is in the correct format before passing to model.
     * @param email String value having email.
     * @param bldName String value for building name.
     * @param room String value for room.
     * @param yyyymmdd String value in yyyy-mm-dd format.
     * @param stTime String value in hh:mm format.
     * @param enTime String value in hh:mm format.
     */
    public void addBooking(String email, String bldName, String room, String yyyymmdd ,String stTime, String enTime) {
        Building building;
        message = "-Timeslot not available.";

        LocalDateTime startTime = getDateTime(yyyymmdd, stTime);
        LocalDateTime endTime = getDateTime(yyyymmdd, enTime);

//        1. This block of time must last at least 5 minutes.
        if (Duration.between(startTime, endTime).toMinutes() < MIN_DURATION) {
            message = "-Min Booking duration should be 5 mins.";
            changed();
            return;
        }

//        2. Booking length must be a multiple of 5 minutes
        if (Duration.between(startTime, endTime).toMinutes() % MIN_DURATION != 0) {
            message = "-Booking length must be a multiple of 5 minutes.";
            changed();
            return;
        }

//        3. check building and room exist
        building = checkBuilding(bldName, room);
        if (building == null) {
            message = "-Add room to the system first.";

        } else if (building.getStartTime().isEmpty()) {  // when no booking exist create new booking.
            createBooking(email, building, startTime, endTime);

        } else {
            Collections.sort(building.getStartTime());
            Collections.sort(building.getEndTime());

            ArrayList<LocalDateTime> stime = new ArrayList<>(building.getStartTime()); // get sorted time.
            ArrayList<LocalDateTime> etime = new ArrayList<>(building.getEndTime()); // get sorted time.

// Note - overlap by 5 minutes or more, then it’s an overlap i.e. overlap up to 4 mins is allowed in booking.
// Adjust time comparison accordingly.
            for (int i = 0; i < etime.size(); i++) {
                boolean booked = false;

                if (endTime.isBefore(stime.get(0).plusMinutes(MIN_DURATION))) {  //check Top of timeslot
                    createBooking(email, building, startTime, endTime);
                    break;

                } else if ((i + 1) >= etime.size()   // check end of timeslot.
                        && startTime.isAfter(etime.get(i).minusMinutes(MIN_DURATION))) {
                        createBooking(email, building, startTime, endTime);

                } else {
                    for (int j = 0; j <= MIN_DURATION; j++) {

                        if (startTime.isAfter(etime.get(i).minusMinutes(MIN_DURATION - j))  // check middle of timeslot.
                                && endTime.isBefore(stime.get(i + 1).plusMinutes(j))) {
                            createBooking(email, building, startTime, endTime);
                            booked = true;
                            break;
                        }
                    }
                    if (booked)
                        break;
                }
            } //for-loop
        }

        changed();
    }

    /**
     * Create a booking object in the system for given details after validation checks.
     * If a person (email) is not in the system it is created at the time of booking.
     * @param email String value for email.
     * @param building Building object.
     * @param startTime LocalDateTime object for start time.
     * @param endTime LocalDateTime object for end time.
     */
    public void createBooking(String email, Building building, LocalDateTime startTime, LocalDateTime endTime) {
        addPerson(email, " ");        // this will add person if doesn't exist.
        Person person = checkPerson(email); // will return person
        Booking booking = new Booking(person, building, startTime, endTime);
        bookingList.add(booking);
        message = "-Booking successful.";
    }

    /**
     * Remove booking from the system after checking existence.
     * UI validated email, date and time is in the correct format before passing to model.
     * @param email String value for email.
     * @param bldgname String value for building.
     * @param room String value for room.
     * @param yyyymmdd String value for date in yyyy-mm-dd.
     * @param stTime String value for start time in hh:mm.
     * @param enTime String value for end time in hh:mm.
     */
    public void removeBooking(String email, String bldgname, String room, String yyyymmdd, String stTime, String enTime) {

        LocalDateTime startTime = getDateTime(yyyymmdd, stTime);
        LocalDateTime endTime = getDateTime(yyyymmdd, enTime);
        message = "-No Booking found.";

        for (Booking bookin: bookingList) {
            if (bookin.getPerson().getEmail().equals(email)
            && bookin.getbuilding().getName().equals(bldgname)
            && bookin.getbuilding().getRoom().equals(room)
            && bookin.getStartTime().isEqual(startTime)
            && bookin.getEndTime().isEqual(endTime)) {
                bookin.getbuilding().freeAllocation(startTime, endTime);
                bookingList.remove(bookin);
                message = "-Booking removed.";
                break;
            }
        }
        changed();
    }

    /**
     * Choose a time, and see all rooms that are available at that time or
     * Choose a timeslot (start and end times) and see all rooms that are available for that whole period.
     * UI validated email, date and time is in the correct format before passing to model.
     * @param stTime String value for start time in hh:mm.
     * @param enTime String value for end time in hh:mm.
     */
    public void showTime(String yyyymmdd, String stTime, String enTime) {

        LocalDateTime startTime = getDateTime(yyyymmdd, stTime);
        LocalDateTime endTime = getDateTime(yyyymmdd, enTime);

        message = "Following room(s) are available:";
        changed();

        for (Building building : buildingList) {

            if (!building.getRoom().equals(" ")
                    && !building.getRoom().equals("")) {

                Collections.sort(building.getStartTime());
                Collections.sort(building.getEndTime());

                ArrayList<LocalDateTime> stime = new ArrayList<>(building.getStartTime()); // get sorted time.
                ArrayList<LocalDateTime> etime = new ArrayList<>(building.getEndTime()); // get sorted time.

//                for no bookings, show the room as available.
                if (stime.isEmpty()) {
                    message = building.getName() + "  " + building.getRoom();
                    changed();
                }

// Note - overlap by 5 minutes or more, then it’s an overlap i.e. overlap of < 5 is allowed in booking.
// Adjust time comparison accordingly. When searching for specific time ( start time = end time or end time not entered)
// set overlap to 0.

                int overlap = startTime.isEqual(endTime) ? 0 : MIN_DURATION;

                for (int i = 0; i < etime.size(); i++) {
                    if (endTime.isBefore(stime.get(0).plusMinutes(overlap))) {  //check Top of timeslot
                        message = building.getName() + "  " + building.getRoom();
                        changed();
                        break;

                    } else if ((i + 1) >= etime.size()   // check end of timeslot.
                            && startTime.isAfter(etime.get(i).minusMinutes(overlap))) {
                        message = building.getName() + "  " + building.getRoom();
                        changed();

                    } else {
                        for (int j = 0; j <= overlap; j++) {

                            if (startTime.isAfter(etime.get(i).minusMinutes(overlap - j))  // check middle of timeslot.
                                    && endTime.isBefore(stime.get(i + 1).plusMinutes(j))) {
                                message = building.getName() + "  " + building.getRoom();
                                changed();
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Checks the schedule for a given room, including bookings and free periods.
     * @param bldgname String value for building name.
     * @param room room String value for room.
     */
    public void showRoom (String bldgname, String room) {

        if (checkBuilding(bldgname, room) == null) {
            message = "Building room not found.";
            changed();
            return;
        }

        for (Building building: buildingList) {

            if (building.getName().equals(bldgname)
                && building.getRoom().equals(room)) {

                Collections.sort(building.getStartTime());
                Collections.sort(building.getEndTime());

                ArrayList<LocalDateTime> stime = new ArrayList<>(building.getStartTime()); // get sorted time.
                ArrayList<LocalDateTime> etime = new ArrayList<>(building.getEndTime()); // get sorted time.


                for (int i = 0; i < etime.size(); i++) {

                    if (i == 0) {
                        message = bldgname.trim() + " " + room.trim() + " :Following slots are booked, rest are available.";
                        changed();
                    }

                    message = "From: " + stime.get(i).toString() + "  " + "To: " + etime.get(i).toString();
                    changed();
                }
            }
        }

    }

    /**
     * checks and return bookings made by a given person.
     * UI validated email is in the correct format before passing to model.
     * @param email String value for email.
     */
    public void showPersonBooking(String email) {
        Boolean found = false;

        if (checkPerson(email) == null) {
            message = "Person not found.";
            changed();
            return;
        }

        message = "Showing bookings for: " + email;
        changed();


        for (Booking booking: bookingList) {

            if (booking.getPerson().getEmail().equals(email)) {

                Formatter fmt = new Formatter();

                message = String.valueOf(fmt.format("%-20s %-10s  %s  %s",booking.getbuilding().getName(),
                        booking.getbuilding().getRoom(), booking.getStartTime().toString(), booking.getEndTime().toString()));

                changed();
                fmt.close();
                found = true;
            }
        }
        if (!found) {
            message = "no booking found.";
            changed();
        }
    }

}
