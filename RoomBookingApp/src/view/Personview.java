package view;

import model.RBSModel;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Personview provides GUI view for Add or Remove of person from RBSModel. It implements RBSModel listener to get updates
 * from model. It also adds itself as a listener to model.
 */

public class Personview implements ActionListener, RBSModel.Listener {
    private final int FRAME_WIDTH = 420;
    private final int FRAME_HEIGHT = 320;

    RBSModel model;
    private JFrame pframe;
    private JButton home;
    private JButton addPerson;
    private JButton removePerson;
    private JLabel email;
    private JLabel name;
    private JLabel message;
    private JTextField emailField;
    private JTextField nameField;

    /**
     * Constructor of Personview
     * @param model RBSModel
     */
    public Personview(RBSModel model) {
        this.model = model;

        pframe = new JFrame("Person view");
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
     * Add buttons and Labels to the frame.
     */
    private void addComponents() {
        home = new JButton("Home");
        addPerson = new JButton("Add");
        removePerson = new JButton("Remove");
        email = new JLabel("*Email");
        name = new JLabel("Name");
        message = new JLabel("*Fields are required.");
        emailField = new JTextField("",220);
        nameField = new JTextField("",220);

        home.setBounds(10, 25, 100, 40); // button position on frame.
        pframe.add(home);

        addPerson.setBounds(110, 25, 100, 40);
        pframe.add(addPerson);

        removePerson.setBounds(210, 25, 100, 40);
        pframe.add(removePerson);

        email.setBounds(10,75, 50, 40);
        pframe.add(email);
        emailField.setBounds(80,75,220,40);
        pframe.add(emailField);

        name.setBounds(10,125,50,40);
        pframe.add(name);
        nameField.setBounds(80,125,220,40);
        pframe.add(nameField);

        message.setBounds(10,200,200,40);
        pframe.add(message);

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
        addPerson.addActionListener(al);
        removePerson.addActionListener(al);
    }

    /**
     * Perform action when button is clicked.
     * @param e the event to be processed
     */

    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == home) {
            pframe.dispose(); // close this frame
            new RBSview(model);    // go home

        } else if (e.getSource() == addPerson) {

            if (!isEmailValid(emailField.getText())) {
                message.setText("Email is required.");
                return;
            }

            model.addPerson(emailField.getText(), nameField.getText());
            resetFields();

        } else if (e.getSource() == removePerson) {

            if (!isEmailValid(emailField.getText())) {
                message.setText("Email is required.");
                return;
            }

            model.removePerson(emailField.getText());
            resetFields();

        }
    }

    /**
     * Reset fields to blanks.
     */
    private void resetFields() {
        emailField.setText("");
        nameField.setText("");
    }

    /**
     * Check for valid email string.
     * @param emailStr String.
     * @return true if valid email else false.
     */
    public boolean isEmailValid(String emailStr) {
        final Pattern VALID_EMAIL_ADDRESS_REGEX = Pattern.compile("^[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,6}$", Pattern.CASE_INSENSITIVE);
        Matcher matcher = VALID_EMAIL_ADDRESS_REGEX.matcher(emailStr);
        return matcher.find();
    }
}
