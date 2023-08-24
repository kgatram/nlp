package main;

import model.RBSModel;
import view.RBSview;

/**
 * This is executable class of Room Booking system (RBS).
 * If previously save model file name or path is pass as an argument during call, it loads the model otherwise
 * it creates a new instance of model. It launches GUI and command line view to interact with model.
 * Both UI runs concurrently in multi-thread allowing the users to interact with model simultaneously.
 * Ending one UI does not terminate other.
 *
 * @author  220025456
 * @version "%I%, %G%"
 * @since JDK17
 */

public class RBSMain {
    /**
     * main method of RBSMain class.
     * @param args String value having filename or path of previously saved model. This is an optional argument.
     */
    public static void main(String[] args) {

        RBSModel model;

//        check if model is to be loaded from file.
        if (args.length == 1) {
            model = new SaveLoadModel().loadModel(args[0]);

            if (model == null) {
                System.out.println("Error Loading model. Exiting RBS");
                System.exit(1);
            }

//      load ok, initialise listener
            model.initialiseListeners();

        } else {
//          if not loading, create a new model.
            model = new RBSModel();
        }

//      Launch GUI
        new RBSview(model);

//      Launch CLI
        CLInterface cli = new CLInterface(model);
        cli.start();
        try {
            cli.join();
        } catch (InterruptedException e) {
            System.out.println("Exiting RBS");
        }

    }
}