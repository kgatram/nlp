package view;

import model.RBSModel;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.regex.Pattern;

/**
 * Roomview provides GUI view for Add or Remove of room/Building from RBSModel. It implements RBSModel listener to get updates
 * from model. It also adds itself as a listener to model.
 */

public class Roomview implements ActionListener, RBSModel.Listener {
    private final int FRAME_WIDTH = 700;
    private final int FRAME_HEIGHT = 320;

    RBSModel model;
    private JFrame roomframe;
    private JButton home;
    private JButton addRoom;
    private JButton removeRoom;
    private JButton addBldg;
    private JButton removeBldg;
    private JLabel room;
    private JLabel building;
    private JLabel address;
    private JLabel message;
    private JTextField roomField;
    private JTextField buildingField;
    private JTextField addressField;

    /**
     * Constructor of Roomview
     * @param model RBSModel
     */
    public Roomview(RBSModel model) {
        this.model = model;

        roomframe = new JFrame("Room view");
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
     * Add Buttons , Fields and Labels to frame.
     */
    private void addComponents() {
        home = new JButton("Home");
        addRoom = new JButton("Add Room");
        removeRoom = new JButton("Rmv Room");
        addBldg = new JButton("Add Building");
        removeBldg = new JButton("Rmv Building");

        room = new JLabel("*Room");
        building = new JLabel("*Building");
        address = new JLabel("Address");
        message = new JLabel("*Fields are required.");

        roomField = new JTextField("",220);
        buildingField = new JTextField("",220);
        addressField = new JTextField("",220);

        home.setBounds(10, 25, 100, 40); // button position on frame.
        roomframe.add(home);

        addRoom.setBounds(110, 25, 125, 40);
        roomframe.add(addRoom);

        removeRoom.setBounds(225, 25, 125, 40);
        roomframe.add(removeRoom);

        addBldg.setBounds(350, 25, 125, 40);
        roomframe.add(addBldg);

        removeBldg.setBounds(475, 25, 125, 40);
        roomframe.add(removeBldg);

        room.setBounds(10,75, 50, 40);
        roomframe.add(room);
        roomField.setBounds(100,75,220,40);
        roomframe.add(roomField);

        building.setBounds(10,125,80,40);
        roomframe.add(building);
        buildingField.setBounds(100,125,220,40);
        roomframe.add(buildingField);

        address.setBounds(10,175,80,40);
        roomframe.add(address);
        addressField.setBounds(100,175,220,40);
        roomframe.add(addressField);

        message.setBounds(10,225,500,40);
        roomframe.add(message);

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
        addRoom.addActionListener(al);
        removeRoom.addActionListener(al);
        addBldg.addActionListener(al);
        removeBldg.addActionListener(al);
    }

    /**
     * Perform action when button is clicked.
     * @param e the event to be processed
     */
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == home) {
            roomframe.dispose();   // close this frame
            new RBSview(model);    // go home

        } else if (e.getSource() == addRoom) {

            if (!Pattern.matches("\\S+", roomField.getText())
                    || !Pattern.matches("\\S+", buildingField.getText())) {

                message.setText("Room and Building is required without spaces.");
                return;
            }

            model.addRoom(roomField.getText(), buildingField.getText(), addressField.getText());
            resetFields();

        } else if (e.getSource() == removeRoom) {

            if (!Pattern.matches("\\S+", roomField.getText())
                    || !Pattern.matches("\\S+", buildingField.getText())) {

                message.setText("Room and Building is required without spaces.");
                return;
            }

            model.removeRoom(roomField.getText(), buildingField.getText());
            resetFields();


        } else if (e.getSource() == addBldg) {

            if (!Pattern.matches("\\S+", buildingField.getText())) {

                message.setText("Building is required without spaces.");
                return;
            }

            model.addBuilding(buildingField.getText(), addressField.getText());
            resetFields();

        } else if (e.getSource() == removeBldg) {

            if (!Pattern.matches("\\S+", buildingField.getText())) {

                message.setText("Building is required without spaces.");
                return;
            }


            model.removeBuilding(buildingField.getText());
            resetFields();
        }
    }

    /**
     * reset fields.
     */

    private void resetFields() {
        roomField.setText("");
        buildingField.setText("");
        addressField.setText("");
    }

}
