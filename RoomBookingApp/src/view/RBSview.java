package view;

import model.RBSModel;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 * This is GUI view of Room Booking System and is the Home screen. It provides navigation options to users to work with
 * Person, Room, Booking and Save the model.
 */

public class RBSview implements ActionListener {

    private final int FRAME_WIDTH = 520;
    private final int FRAME_HEIGHT = 200;
    RBSModel model;
    private JFrame rbsframe;

    private JButton person;
    private JButton room;
    private JButton booking;
    private JButton save;

    /**
     * Constructor of RBSview.
     * @param model RBSModel
     */
    public RBSview(RBSModel model)  {

        this.model = model;

        rbsframe = new JFrame("Room Booking System");
        rbsframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        rbsframe.setSize(FRAME_WIDTH, FRAME_HEIGHT); // set frame size
        rbsframe.setLayout(null);
        rbsframe.setVisible(true); // display frame

        addHomeComponents();
        addActionListenerForButtons(this);
    }

    /**
     * Add buttons and Labels to the frame.
     */
    private void addHomeComponents() {
        person = new JButton("Person");
        room = new JButton("Room");
        booking = new JButton("Booking");
        save = new JButton("Save");

        person.setBounds(10, 25, 100, 40); // button position on frame.
        rbsframe.add(person);

        room.setBounds(110, 25, 100, 40);
        rbsframe.add(room);

        booking.setBounds(210, 25, 100, 40);
        rbsframe.add(booking);

        save.setBounds(310, 25, 100, 40);
        rbsframe.add(save);
    }

    /**
     * Listens to button for click.
     * @param al ActionListener
     */
    public void addActionListenerForButtons(ActionListener al) {
        person.addActionListener(al);
        room.addActionListener(al);
        booking.addActionListener(al);
        save.addActionListener(al);
    }

    /**
     * Perform action when button is clicked.
     * @param e the event to be processed
     */

    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == person) {
            rbsframe.dispose(); // close this frame
            new Personview(model);   // go to person view

        } else if (e.getSource() == room) {
            rbsframe.dispose();
            new Roomview(model);

        } else if (e.getSource() == booking) {
            rbsframe.dispose();
            new Bookingview(model);

        } else if (e.getSource() == save) {
            rbsframe.dispose();
            new Saveview(model);
        }


    }

}
