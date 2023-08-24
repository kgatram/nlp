package view;


import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.util.Vector;

/**
 * Table view display model information in tabular form.
 */

public class Tableview  {

    private final int FRAME_WIDTH = 500;
    private final int FRAME_HEIGHT = 300;
    private JFrame frame;

    private JTable table;
    Vector<String> record;


    /**
     * Constructor for Tableview
     * @param record a vector of strings containing all the records to display.
     */
    public Tableview(Vector<String> record)  {
        this.record = record;

        frame = new JFrame("Table view");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(FRAME_WIDTH, FRAME_HEIGHT); // set frame size
        frame.setVisible(true); // display frame

        addHomeComponents();
    }

    /**
     * Add Table to Frame.
     */
    private void addHomeComponents() {
        // Column heading
        Vector<String> heading = new Vector<>();
        heading.addElement("Booking details");

        // Initializing the JTable
        table = new JTable();
        table.setBounds(30, 40, 200, 300);

        DefaultTableModel model = new DefaultTableModel();
        model.setColumnIdentifiers(heading);

        for (String row : record) {
            Vector<String> rec = new Vector<>();
            rec.addElement(row);
            model.addRow(rec);
        }

        table.setModel(model);
        JScrollPane sp = new JScrollPane(table);
        frame.add(sp);
    }


}
