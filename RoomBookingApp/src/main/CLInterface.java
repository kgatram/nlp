package main;

import model.RBSModel;
import model.RBSModel.Listener;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.LocalDateTime;
import java.time.format.DateTimeParseException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * This class is view part of Model-Delegate design pattern and provides text based command-line interface (CLI) to Room Booking system
 * (RBS) hence called CLInterface. This class implements RBSModel Listener interface by implementing update method and add itself as
 * listener to get updates from RBSModel. (Reference Frog example.). Any changes in the RBSModel is communicated to this class via update
 * method.
 * RBSMain execute instance of this class in a child thread allowing concurrent availability of this view along with other views of RBS.
 * This class provide various commands that users can enter on terminal command line to interact with model. Any information update from
 * model is reflected back on to the terminal. Users can get list of available command by typing HELP command. Typing a command without any
 * argument will show usuage of command.
 *
 * @author  220025456
 * @version "%I%, %G%"
 * @since JDK17
 */
public class CLInterface extends Thread implements Listener {
    private BufferedReader reader;
    private final Pattern space = Pattern.compile(" ");
    RBSModel model;

    public CLInterface (RBSModel model) {
        this.model = model;
         reader = new BufferedReader(new InputStreamReader(System.in));

        // Register as a listener so that the model notifies us of changes
        model.addListener(this);
    }

    /**
     * Whenever there's change in model, it triggers this method.
     */
    public void update() {
        System.out.println(model.getMessage());
    }

    /**
     * initiate a thread for CLInterface object.
     *
     */
    public void run() {
        try {
            executeThread();
        } catch (Exception exp) {
            stopThread();
        }
    }

    /**
     * Handles all the command input from terminal. QUIT command throws IOException to stop thread execution and end terminal session
     * @throws IOException signals thread to stop.
     */

    private void executeThread() throws IOException {
        String data;
        String[] cmdstr;

        while (true) {
            System.out.println("*args are required. Please enter a command:");
            data = reader.readLine();
            cmdstr = space.split(data, 2);

            switch (cmdstr[0]) {
                case "ADDPER", "RMVPER" -> wrkPerson(data);
                case "ADDROM", "RMVROM" -> wrkRoom(data);
                case "ADDBLD", "RMVBLD" -> wrkBuilding(data);
                case "ADDBOK", "RMVBOK" -> wrkBooking(data);
                case "SHWROM" -> showRoom(data);
                case "SHWTIM" -> showTime(data);
                case "SHWPER" -> showPerson(data);
                case "SAVE" -> saveModel(data);
                case "HELP" -> {
                    System.out.println("Available commands: ADDPER RMVPER ADDROM RMVROM ADDBLD RMVBLD ADDBOK RMVBOK SHWROM SHWTIM SHWPER SAVE HELP QUIT");
                }
                case "QUIT" -> {
                    System.out.println("Session Ended.");
                    throw new IOException();
                }
                default -> {
                }
            }
        }
    }

    /**
     * Close input reader and end terminal session.
     */
    private void stopThread() {
        try {
            reader.close();
        } catch (Exception exp) {
            System.out.println("Thread stopped.");
        }

    }

    /**
     * Checks for valid email pattern in the string.
     * @param emailStr String value having email
     * @return true if valid email format else false.
     */
    public boolean isEmailValid(String emailStr) {
        final Pattern VALID_EMAIL_ADDRESS_REGEX = Pattern.compile("^[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,6}$", Pattern.CASE_INSENSITIVE);
        Matcher matcher = VALID_EMAIL_ADDRESS_REGEX.matcher(emailStr);
        return matcher.find();
    }
    /**
     * ADDPER - Add or RMVPER - Remove person by calling appropriate model methods.
     * @param input String value containing command email name separated by space. space is delimiter so command and email cannot have
     * space, name can have spaces.
     */
    private void wrkPerson (String input) {
        String[] str = space.split(input, 3);
        String email = str.length >= 2 ? str[1] : " ";
        String name  = str.length == 3 ? str[2] : " " ;

        if (email.equals(" ")) {
            System.out.println("Usage: " + str[0] + " *email name");
            return;
        }

        if (!isEmailValid(email)) {
            System.out.println("Invalid email.");
            return;
        }

        if (str[0].equals("ADDPER")) {
            model.addPerson(email, name);
        } else {
            model.removePerson(email);
        }
    }


    /**
     * ADDROM - Add or RMVROM - Remove room by calling appropriate model methods.
     * @param input String value containing command room building address separated by space. Since space is delimiter, command, room,
     * building cannot have spaces. Address can have spaces. Room and Building name is required parameters whereas Address can optional.
     */
    private void wrkRoom (String input) {
        String[] str = space.split(input, 4);
        String room     = str.length >= 2 ? str[1] : " ";
        String building = str.length >= 3 ? str[2] : " ";
        String address  = str.length == 4 ? str[3] : " ";

        if (room.equals(" ")
            || building.equals(" ")) {
            System.out.println("Usage: " + str[0] + " *room *building address");
            return;
        }

        if (str[0].equals("ADDROM")) {
            model.addRoom(room, building, address);
        } else {
            model.removeRoom(room, building);
        }
    }


    /**
     * ADDBLD - Add or RMVBLD - Remove Building by calling appropriate model methods.
     * @param input String value containing command building address separated by space. Since space is delimiter, command,
     * building cannot have spaces. Address can have spaces. Building name is required parameters whereas Address can optional.
     */
    private void wrkBuilding (String input) {
        String[] str = space.split(input, 3);
        String building = str.length >= 2 ? str[1] : " ";
        String address  = str.length == 3 ? str[2] : " ";

        if (building.equals(" ")) {
            System.out.println("Usage: " + str[0] + " *building address");
            return;
        }

        if (str[0].equals("ADDBLD")) {
            model.addBuilding(building, address);
        } else {
            model.removeBuilding(building);
        }
    }


    /**
     * ADDBOK - Add or RMVBOK - Remove Booking by calling appropriate model methods.
     * @param input String value containing command, email, building, room, date, starttime, endtime. Since space is delimiter, command,
     * room, building cannot have spaces. Date string must be in yyyy-mm-dd format. time must be in hh:mm format. All are required arguments.
     */
    private void wrkBooking (String input) {
        String[] str = space.split(input, 7);
        String email      = str.length >= 2 ? str[1] : " ";
        String building   = str.length >= 3 ? str[2] : " ";
        String room       = str.length >= 4 ? str[3] : " ";
        String yyyymmdd   = str.length >= 5 ? str[4] : " ";
        String starttime  = str.length >= 6 ? str[5] : " " ;
        String endtime    = str.length == 7 ? str[6] : " " ;

        if ( email.equals(" ")
            || building.equals(" ")
            || room.equals(" ")
            || yyyymmdd.equals(" ")
            || starttime.equals(" ")
            || endtime.equals(" ")) {

            System.out.println("Usage: " + str[0] + " *email *buidling *room *date *starttime *endtime");
            return;
        }

        if (!isEmailValid(email)) {
            System.out.println("Incorrect email format.");
            return;
        }

        starttime = formatTime(starttime);
        endtime   = formatTime(endtime);

        if (!validTime(starttime)
                || !validTime(endtime)) {
            System.out.println("Time required in hh:mm");
            return;
        }

        if (!validDate(yyyymmdd, starttime)) {
            System.out.println("Valid Datetime in future required. Date as yyyy-mm-dd time as hh:mm.");
            return;
        }

        if (str[0].equals("ADDBOK")) {
            model.addBooking(email, building, room, yyyymmdd, starttime, endtime);
        } else {
            model.removeBooking(email, building, room, yyyymmdd, starttime, endtime);
        }
    }

    /**
     * Checks time input is valid time.
     * @param hhmm string value in hh:mm format.
     * @return true if valid time else false.
     */
    public boolean validTime (String hhmm) {
        return Pattern.matches("^([01]?[0-9]|2[0-3]):[0-5][0-9]$", hhmm);
    }

    /**
     * Check input date and time string is valid date and is in future.
     * @param date in yyyy-mm-dd format
     * @param hhmm in hh:mm format
     * @return true if valid datetime or else false.
     */
    public boolean validDate (String date, String hhmm) {
        try {
            return LocalDateTime.parse(date + "T" + hhmm).isAfter(LocalDateTime.now());
        } catch (DateTimeParseException dtpe) {
            return false;
        }
    }

    /**
     * Formats a time if given value is in h:mm
     * @param timestring time in h:mm
     * @return string in hh:mm
     */
    public String formatTime (String timestring) {
        return Pattern.matches("[0-9]{2}:[0-9]{2}", timestring) ? timestring : "0" + timestring;
    }

    /**
     * SHWROM - Execute model method to show schedule for a given room, including bookings and free periods.
     * @param input String value containing command room building separated by space. Since space is delimiter, command, room,
     * building cannot have spaces. Room and Building name is required parameters.
     */
    private void showRoom (String input) {
        String[] str = space.split(input, 3);
        String room     = str.length >= 2 ? str[1] : " ";
        String building = str.length >= 3 ? str[2] : " ";

        if (building.equals(" ")
                || room.equals(" ")) {
            System.out.println("Usage: SHWROM *room *building ");
            return;
        }

        model.showRoom(building, room);
    }

    /**
     * SHWTIM - Choose a time, and see all rooms that are available at that time or
     * Choose a timeslot (start and end times) and see all rooms that are available for that whole period.
     * @param input String value containing command, date, starttime, endtime.
     * Date string must be in yyyy-mm-dd format. time must be in hh:mm format. All are required arguments.
     */
    private void showTime(String input) {
        String[] str = space.split(input, 3);
        String yyyymmdd    = str.length >= 2 ? str[1] : " ";
        String startTime   = str.length >= 3 ? str[2] : " ";
        String endTime     = str.length == 4 ? str[3] : startTime;

        startTime = formatTime(startTime);
        endTime   = formatTime(endTime);

        if (!validTime(startTime)
                || !validTime(endTime)) {
            System.out.println("Time required in hh:mm");
            System.out.println("Usage: SHWTIM *date *starttime endtime");
            return;
        }

        if (!validDate(yyyymmdd, startTime)) {
            System.out.println("Valid Date required in yyyy-mm-dd.");
            System.out.println("Usage: SHWTIM *date *starttime endtime");
            return;
        }

        model.showTime(yyyymmdd, startTime, endTime);
    }

    /**
     * SHWPER - Show a person's booking.
     * @param input String value containing command and email. email is required in correct format.
     *
     * */
    private void showPerson(String input) {
        String[] str = space.split(input, 2);
        String email   = str.length == 2 ? str[1] : " ";

        if (email.equals(" ")) {
            System.out.println("Usage: SHWPER *email");
            return;
        }

        if (!isEmailValid(email)) {
            System.out.println("Invalid email.");
            return;
        }

        model.showPersonBooking(email);
    }

    /**
     * SAVE - Save model to a file.
     * @param input String containing command filename. Filename is required.
     */

    private void saveModel(String input) {
        String[] str = space.split(input, 2);
        String filename   = str.length == 2 ? str[1] : " ";

        if (filename.equals(" ")) {
            System.out.println("Usage: SAVE *filename");
            return;
        }

        if (!Pattern.matches("\\S+", filename)) {
            System.out.println("file path is required without spaces.");
            return;
        }

        new SaveLoadModel().saveModel(model, filename);
    }

}
