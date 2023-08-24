package view;

import model.RBSModel;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 * Showbooking presents navigation options to the users to select booking by Person, Room or Time.
 */

public class Showbooking implements ActionListener {

    private final int FRAME_WIDTH = 600;
    private final int FRAME_HEIGHT = 200;
    RBSModel model;
    private JFrame sframe;

    private JButton person, back;
    private JButton room;
    private JButton time;

    /**
     * Constructor for Showbooking.
     * @param model RBSModel
     */

    public Showbooking(RBSModel model)  {

        this.model = model;

        sframe = new JFrame("Show Booking");
        sframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        sframe.setSize(FRAME_WIDTH, FRAME_HEIGHT); // set frame size
        sframe.setLayout(null);
        sframe.setVisible(true); // display frame

        addHomeComponents();
        addActionListenerForButtons(this);
    }

    /**
     * Add Buttons and Labels to field.
     */
    private void addHomeComponents() {
        back = new JButton("Back");
        person = new JButton("Person");
        room = new JButton("Room");
        time = new JButton("Time");

        back.setBounds(10, 25, 100, 40); // button position on frame.
        sframe.add(back);

        person.setBounds(110, 25, 100, 40); // button position on frame.
        sframe.add(person);

        room.setBounds(210, 25, 100, 40);
        sframe.add(room);

        time.setBounds(310, 25, 100, 40);
        sframe.add(time);
    }

    /**
     * Listens to button for click.
     * @param al ActionListener
     */
    public void addActionListenerForButtons(ActionListener al) {
        back.addActionListener(al);
        person.addActionListener(al);
        room.addActionListener(al);
        time.addActionListener(al);
    }

    /**
     * Perform action when button is clicked.
     * @param e the event to be processed
     */

    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == back) {
            sframe.dispose(); // close this frame
            new Bookingview(model);   // go back to Booking view.

        } else if (e.getSource() == room) {
            sframe.dispose();
            new Roombooking(model);

        } else if (e.getSource() == person) {
            sframe.dispose();
            new Personbooking(model);

        } else if (e.getSource() == time) {
            sframe.dispose();
            new Timebooking(model);
        }
    }

}
