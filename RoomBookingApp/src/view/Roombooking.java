package view;

import model.RBSModel;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Vector;
import java.util.regex.Pattern;

/**
 * Room booking present GUI for users to query RBSModel to show bookings for a given room. It implements RBSModel listener to get
 * updates from model. It also adds itself as a listener to the model.
 */
public class Roombooking implements ActionListener, RBSModel.Listener {
    private final int FRAME_WIDTH = 700;
    private final int FRAME_HEIGHT = 320;

    RBSModel model;
    private JFrame roomframe;
    private JButton back;
    private JButton search;
    private JLabel room;
    private JLabel building;
    private JLabel message;
    private JTextField roomField;
    private JTextField buildingField;

    Vector<String> record;

    /**
     * Constructor for Roombooking.
     * @param model RBSModel
     */
    public Roombooking(RBSModel model) {
        this.model = model;
        record = new Vector<>();

        roomframe = new JFrame("Room Booking Search");
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

        room = new JLabel("*Room");
        building = new JLabel("*Building");
        message = new JLabel("*Fields are required.");

        roomField = new JTextField("",220);
        buildingField = new JTextField("",220);

        back.setBounds(10, 25, 100, 40); // button position on frame.
        roomframe.add(back);

        search.setBounds(110, 25, 100, 40);
        roomframe.add(search);


        room.setBounds(10,75, 50, 40);
        roomframe.add(room);
        roomField.setBounds(80,75,220,40);
        roomframe.add(roomField);

        building.setBounds(10,125,80,40);
        roomframe.add(building);
        buildingField.setBounds(80,125,220,40);
        roomframe.add(buildingField);


        message.setBounds(10,225,500,40);
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

            if (!Pattern.matches("\\S+", roomField.getText())
                    || !Pattern.matches("\\S+", buildingField.getText())) {

                message.setText("Room and Building is required without spaces.");
                return;
            }

            model.showRoom(buildingField.getText(), roomField.getText());
            new Tableview(record);
            resetFields();
        }
    }

    /**
     * reset fields.
     */
    private void resetFields() {
        message.setText("Info for: " + roomField.getText() + " " + buildingField.getText());
        roomField.setText("");
        buildingField.setText("");
        record.clear();
    }

}
