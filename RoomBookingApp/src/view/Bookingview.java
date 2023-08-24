package view;

import model.RBSModel;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeParseException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Bookingview provides GUI view to Add, Remove and Show bookings to RBSModel. It implements RBSModel listener to get updates
 * from model. It also adds itself as a listener to the model.
 */

public class Bookingview implements ActionListener, RBSModel.Listener {
    private final int FRAME_WIDTH = 700;
    private final int FRAME_HEIGHT = 500;

    RBSModel model;
    private JFrame bframe;
    private JButton home;
    private JButton addBooking;
    private JButton removeBooking;
    private JButton showBooking;
    private JLabel room;
    private JLabel building;
    private JLabel email;
    private JLabel stime, date;
    private JLabel etime;
    private JLabel message;
    private JTextField roomField;
    private JTextField buildingField;
    private JTextField emailField;
    private JTextField stimeField, dateField;
    private JTextField etimeField;

    /**
     * Constructor for Bookingview
     * @param model RBSModel
     */
    public Bookingview(RBSModel model) {
        this.model = model;

        bframe = new JFrame("Booking view");
        bframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        bframe.setSize(FRAME_WIDTH, FRAME_HEIGHT); // set frame size
        bframe.setLayout(null);
        bframe.setVisible(true); // display frame

        addComponents();
        addActionListenerForButtons(this);

//        register to model.
        model.addListener(this);
    }

    /**
     * Add Buttons, Fields, Labels to Frame.
     */

    private void addComponents() {
        home = new JButton("Home");
        addBooking = new JButton("Add Booking");
        removeBooking = new JButton("Rmv Booking");
        showBooking = new JButton("Shw Booking");

        email = new JLabel("*Email");
        room = new JLabel("*Room");
        building = new JLabel("*Building");
        date = new JLabel("*Date (yyyy-mm-dd)");
        stime = new JLabel("*Start Time (hh:mm)");
        etime = new JLabel("*End Time (hh:mm)");
        message = new JLabel("*Fields are required.");

        emailField = new JTextField("",220);
        roomField = new JTextField("",220);
        buildingField = new JTextField("",220);
        dateField = new JTextField("",100);
        stimeField = new JTextField("",100);
        etimeField = new JTextField("",100);

        home.setBounds(10, 25, 100, 40); // button position on frame.
        bframe.add(home);

        addBooking.setBounds(110, 25, 125, 40);
        bframe.add(addBooking);

        removeBooking.setBounds(235, 25, 125, 40);
        bframe.add(removeBooking);

        showBooking.setBounds(355, 25, 125, 40);
        bframe.add(showBooking);

// email label and text
        email.setBounds(10,75, 50, 40);
        bframe.add(email);
        emailField.setBounds(175,75,220,40);
        bframe.add(emailField);

// room label and textfield
        room.setBounds(10,125, 50, 40);
        bframe.add(room);
        roomField.setBounds(175,125,220,40);
        bframe.add(roomField);

// building label and textfield
        building.setBounds(10,175,80,40);
        bframe.add(building);
        buildingField.setBounds(175,175,220,40);
        bframe.add(buildingField);

// date label and textfield
        date.setBounds(10,225,150,40);
        bframe.add(date);
        dateField.setBounds(175,225,100,40);
        bframe.add(dateField);
        dateField.setText(LocalDate.now().toString());

// start time label and textfield
        stime.setBounds(10,275,150,40);
        bframe.add(stime);
        stimeField.setBounds(175,275,100,40);
        bframe.add(stimeField);
        stimeField.setText(LocalTime.now().plusMinutes(30).toString().substring(0,5));

// end time label and textfield
        etime.setBounds(10,325,150,40);
        bframe.add(etime);
        etimeField.setBounds(175,325,100,40);
        bframe.add(etimeField);
        etimeField.setText(LocalTime.now().plusMinutes(60).toString().substring(0,5));

        message.setBounds(10,375,500,40);
        bframe.add(message);

    }

    /**
     * Whenever there's change in model, it triggers this method.
     */
    public void update() {
        message.setText(model.getMessage());
    }

    /**
     * Listens to button for click.
     * @param al ActionListener
     */
    public void addActionListenerForButtons(ActionListener al) {
        home.addActionListener(al);
        addBooking.addActionListener(al);
        removeBooking.addActionListener(al);
        showBooking.addActionListener(al);
    }

    /**
     * Perform action when button is clicked.
     * @param e the event to be processed
     */

    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == home) {
            bframe.dispose();   // close this frame
            new RBSview(model);    // go home


        } else if (e.getSource() == addBooking) {
            doBooking("ADD");

        } else if (e.getSource() == removeBooking) {
            doBooking("REMOVE");

        } else if (e.getSource() == showBooking) {
            bframe.dispose();   // close this frame
            new Showbooking(model);
        }
    }

    /**
     * Add or Remove booking from model.
     * @param action Add or remove.
     */
    private void doBooking (String action) {

        if (!isEmailValid(emailField.getText())) {
            message.setText("Incorrect email format.");
            return;
        }

        if (!Pattern.matches("\\S+", roomField.getText())
                || !Pattern.matches("\\S+", buildingField.getText())) {

            message.setText("Room and Building is required without spaces.");
            return;
        }

        String startTime = formatTime(stimeField.getText());
        String endTime   = formatTime(etimeField.getText());

        if (!validTime(startTime)
                || !validTime(endTime)) {
            message.setText("Incorrect time.");
            return;
        }

        if (!validDate(dateField.getText(), startTime)) {
            message.setText("Valid Date and time in future required.");
            return;
        }

        if (action.equals("REMOVE")) {
            model.removeBooking(emailField.getText(), buildingField.getText(), roomField.getText(),
                     dateField.getText(), startTime, endTime);
        } else {
            model.addBooking(emailField.getText(), buildingField.getText(), roomField.getText(),
                     dateField.getText(), startTime, endTime);
        }

        resetFields();
    }

    /**
     * reset fields.
     */
    private void resetFields() {
        emailField.setText("");
        roomField.setText("");
        buildingField.setText("");
        stimeField.setText("");
        etimeField.setText("");
        dateField.setText(LocalDate.now().toString());
        stimeField.setText(LocalTime.now().plusMinutes(30).toString().substring(0,5));
        etimeField.setText(LocalTime.now().plusMinutes(60).toString().substring(0,5));
    }

    /**
     * Check for valid email string.
     * @param emailStr String
     * @return true if valid email else false.
     */
    public boolean isEmailValid(String emailStr) {
        final Pattern VALID_EMAIL_ADDRESS_REGEX = Pattern.compile("^[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,6}$", Pattern.CASE_INSENSITIVE);
        Matcher matcher = VALID_EMAIL_ADDRESS_REGEX.matcher(emailStr);
        return matcher.find();
    }

    /**
     * Format time if given in h:mm to hh:mm
     * @param timestring string
     * @return string in hh:mm
     */
    public String formatTime (String timestring) {
        return Pattern.matches("[0-9]{2}:[0-9]{2}", timestring) ? timestring : "0" + timestring;
    }

    /**
     * Check if given string is valid hh:mm time.
     * @param hhmm string
     * @return true if valid hh:mm else false
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


}
