package view;

import model.RBSModel;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.lang.reflect.GenericDeclaration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeParseException;
import java.util.Vector;
import java.util.regex.Pattern;

/**
 * Time booking present GUI for users to query RBSModel to show bookings for a given time (specific time or time range). It implements RBSModel listener to get
 * updates from model. It also adds itself as a listener to the model.
 */
public class Timebooking implements ActionListener, RBSModel.Listener {
    private final int FRAME_WIDTH = 500;
    private final int FRAME_HEIGHT = 350;

    RBSModel model;
    private JFrame roomframe;
    private JButton back;
    private JButton search;
    private JLabel stime, date;
    private JLabel etime;
    private JLabel message;
    private JTextField stimeField, dateField;
    private JTextField etimeField;

    Vector<String> record;

    /**
     * Constructor for Timebooking.
     * @param model RBSModel
     */
    public Timebooking(RBSModel model) {
        this.model = model;
        record = new Vector<>();

        roomframe = new JFrame("Time Booking Search");
        roomframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        roomframe.setSize(FRAME_WIDTH, FRAME_HEIGHT); // set frame size
        roomframe.setLayout(null);
        roomframe.setVisible(true); // display frame

        addComponents();
        addActionListenerForButtons(this);

//        register to model.
        model.addListener(this);
    }

    /**
     * Add Button, Labels, Fields to Frame.
     */
    private void addComponents() {
        back = new JButton("Back");
        search = new JButton("Search");

        date = new JLabel("*Date in yyyy-mm-dd");
        stime = new JLabel("*Start time (hh:mm)");
        etime = new JLabel("End time (hh:mm)");
        message = new JLabel("*Fields are required.");

        dateField = new JTextField("",220);
        stimeField = new JTextField("",220);
        etimeField = new JTextField("",220);

        back.setBounds(10, 25, 100, 40); // button position on frame.
        roomframe.add(back);

        search.setBounds(110, 25, 100, 40);
        roomframe.add(search);

        date.setBounds(10,75, 150, 40);
        roomframe.add(date);
        dateField.setBounds(175,75,100,40);
        roomframe.add(dateField);
        dateField.setText(LocalDate.now().toString());

        stime.setBounds(10,125, 150, 40);
        roomframe.add(stime);
        stimeField.setBounds(175,125,100,40);
        roomframe.add(stimeField);
        stimeField.setText(LocalTime.now().plusMinutes(30).toString().substring(0,5));


        etime.setBounds(10,175,150,40);
        roomframe.add(etime);
        etimeField.setBounds(175,175,100,40);
        roomframe.add(etimeField);
        etimeField.setText(LocalTime.now().plusMinutes(60).toString().substring(0,5));

        message.setBounds(10,225,300,40);
        roomframe.add(message);

    }

    /**
     * Whenever there's change in model, it triggers this method.
     * method collects all the messages from model in string vector.
     */
    public void update() {
        String row = model.getMessage();
        if (!Pattern.matches("^-.*", row)) {
            record.addElement(row);
        }
    }

    /**
     * Listens to button for click.
     * @param al ActionListener
     */
    public void addActionListenerForButtons(ActionListener al) {
        back.addActionListener(al);
        search.addActionListener(al);
    }

    /**
     * Perform action when button is clicked.
     * @param e the event to be processed
     */
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == back) {
            roomframe.dispose();   // close this frame
            new Showbooking(model);    // go back


        } else if (e.getSource() == search) {
            String endTime;

            endTime = etimeField.getText();

            // set end time = start time when not keyed.
            if (endTime.equals("")
                    || endTime.equals(" ")) {
                endTime = stimeField.getText();
            }

            String startTime = formatTime(stimeField.getText());
                   endTime   = formatTime(endTime);

            if (!validTime(startTime)
                    || !validTime(endTime)) {
                message.setText("Incorrect time.");
                return;
            }

            if (!validDate(dateField.getText(), startTime)) {
                message.setText("Valid Date and time is required.");
                return;
            }

            model.showTime(dateField.getText(), startTime, endTime);
            new Tableview(record);
            message.setText("Info for: " + dateField.getText() + " " + startTime + " " + endTime);
            resetFields();

        }
    }

    /**
     * Reset fields.
     */
    private void resetFields() {
        dateField.setText(LocalDate.now().toString());
        stimeField.setText(LocalTime.now().plusMinutes(30).toString().substring(0,5));
        etimeField.setText(LocalTime.now().plusMinutes(60).toString().substring(0,5));
        record.clear();
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
            LocalDateTime.parse(date + "T" + hhmm);
        } catch (DateTimeParseException dtpe) {
            return false;
        }
        return true;
    }

}
