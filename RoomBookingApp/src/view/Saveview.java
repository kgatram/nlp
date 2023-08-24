package view;

import main.SaveLoadModel;
import model.RBSModel;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.regex.Pattern;

/**
 * Saveview provides GUI view to save RBSModel to a file.
 *
 */

public class Saveview implements ActionListener {
    private final int FRAME_WIDTH = 420;
    private final int FRAME_HEIGHT = 200;

    RBSModel model;
    private JFrame pframe;
    private JButton home;
    private JButton saveButton;
    private JLabel fileLabel;
    private JLabel message;
    private JTextField fileField;

    /**
     * Constructor for Saveview.
     * @param model RBSModel.
     */
    public Saveview(RBSModel model) {
        this.model = model;

        pframe = new JFrame("Save Model");
        pframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        pframe.setSize(FRAME_WIDTH, FRAME_HEIGHT); // set frame size
        pframe.setLayout(null);
        pframe.setVisible(true); // display frame

        addComponents();
        addActionListenerForButtons(this);

    }

    /**
     * Add Button, Field, labels to frame.
     */
    private void addComponents() {
        home = new JButton("Home");
        saveButton = new JButton("Save");
        fileLabel = new JLabel("*File name");
        message = new JLabel();
        fileField = new JTextField("",220);

        home.setBounds(10, 25, 100, 40); // button position on frame.
        pframe.add(home);

        saveButton.setBounds(110, 25, 100, 40);
        pframe.add(saveButton);

        fileLabel.setBounds(10,75, 100, 40);
        pframe.add(fileLabel);
        fileField.setBounds(110,75,220,40);
        pframe.add(fileField);

        message.setBounds(10,125,350,40);
        pframe.add(message);

    }

    /**
     * Add listener to button click.
     * @param al ActionListener
     */
    public void addActionListenerForButtons(ActionListener al) {
        home.addActionListener(al);
        saveButton.addActionListener(al);
    }

    /**
     * Perform action on button click
     * @param e the event to be processed
     */

    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == home) {
            pframe.dispose(); // close this frame
            new RBSview(model);    // go home

        } else if (e.getSource() == saveButton) {

            if (!Pattern.matches("\\S+", fileField.getText())) {

                message.setText("File path is required without spaces.");
                return;
            }

            new SaveLoadModel().saveModel(model, fileField.getText());

            resetFields();

        }
    }

    /**
     * reset fields.
     */
    private void resetFields() {
        message.setText("Saving " + fileField.getText());
        fileField.setText("");
    }

}
