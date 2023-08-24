package main;

import model.RBSModel;

import java.io.*;

/**
 * This class provides functionality to write or save a model object into a file and also load or read a model back onto RBS.
 */

public class SaveLoadModel {

    /**
     * Save a model into a file.
     * @param model RBS model object is required.
     * @param filename String containing filename or path.
     */
    public void saveModel(RBSModel model, String filename) {
        File file = new File(filename);
        ObjectOutputStream oos = null;

        try {
            oos = new ObjectOutputStream(new FileOutputStream(file));
            oos.writeObject(model);

        } catch (IOException ioe) {
            System.out.println(ioe.getMessage());

        } finally {
            try {
                if (oos != null) oos.close();
            } catch (IOException e) {
                System.out.println("Error closing output stream.");
            }
        }
    }

    /**
     * Returns RBSModel object from a saved location
     * @param filename String containing path name
     * @return RBSModel object.
     */

    public RBSModel loadModel (String filename) {
        ObjectInputStream ois = null;
        RBSModel model;
        File file = new File(filename);
        try {
             ois = new ObjectInputStream(new FileInputStream(file));
             model = (RBSModel) ois.readObject();
        } catch (Exception e) {
            return null;
        } finally {
            try {
                if (ois != null) ois.close();
            } catch (IOException e) {
                System.out.println("Error closing input stream.");
            }
        }

        return model;
    }

}
