package view;

import model.RBSModel;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Person booking present GUI for users to query RBSModel to show bookings for a given person. It implements RBSModel listener to get
 * updates from model. It also adds itself as a listener to the model.
 */

public class Personbooking implements ActionListener, RBSModel.Listener {
    private final int FRAME_WIDTH = 420;
    private final int FRAME_HEIGHT = 200;

    RBSModel model;
    private JFrame pframe;
    private JButton back;
    private JButton search;
    private JLabel email;
    private JLabel message;
    private JTextField emailField;
    Vector<String> record;

    /**
     * Constructor for Personbooking.
     * @param model RBSModel
     */
    public Personbooking(RBSModel model) {
        this.model = model;

        record = new Vector<>();

        pframe = new JFrame("Person Booking Search");
        pframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        pframe.setSize(FRAME_WIDTH, FRAME_HEIGHT); // set frame size
        pframe.setLayout(null);
        pframe.setVisible(true); // display frame

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
        email = new JLabel("*Email");
        message = new JLabel("*Fields are required.");
        emailField = new JTextField("",220);

        back.setBounds(10, 25, 100, 40); // button position on frame.
        pframe.add(back);

        search.setBounds(110, 25, 100, 40);
        pframe.add(search);

        email.setBounds(10,75, 50, 40);
        pframe.add(email);
        emailField.setBounds(80,75,220,40);
        pframe.add(emailField);

        message.setBounds(10,125,200,40);
        pframe.add(message);

    }

    /**
     * Whenever there's change in model, it triggers this method.
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
            pframe.dispose(); // close this frame
            new Showbooking(model);    // go back

        } else if (e.getSource() == search) {

            if (!isEmailValid(emailField.getText())) {
                message.setText("Email is required.");
                return;
            }

            model.showPersonBooking(emailField.getText());
            new Tableview(record);
            resetFields();

        }
    }

    /**
     * Reset fields.
     */
    private void resetFields() {
        message.setText("Info for: " + emailField.getText());
        emailField.setText("");
        record.clear();
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
}
