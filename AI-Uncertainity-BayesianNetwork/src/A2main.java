import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

import static java.lang.System.exit;

/********************
 * Starter Code
 *
 * Main driver class to run the application.
 *  Usage:     java A2main <Pn> <NID>
 * Example:   java A2main P1 ~/BNA.xml
 *
 * P1: elimination order is set to the default order
 * P2: elimination order is set to the user input order
 * P3: elimination order is algorithmically determined by reverse topological sort.
 *
 *compile as: ~/Documents/CS5011/A2 $ javac src/*.java
 * @author lf28, 220025456
 *
 */

public class A2main {

    public static void main(String[] args) {

        BayesianNetwork bn = null;

        try {
            bn = NetworkFactory.networkBuilder(args[1]);
            // check for valid cpts.
            bn.validate();
        } catch (ArrayIndexOutOfBoundsException aioobe) {
            System.out.println("Usage: java A2main <Pn> <NID>");
            exit(1);
        } catch (IllegalArgumentException iae) {
            System.out.println(iae.getMessage());
            exit(1);
        } catch (Exception e) {
            e.printStackTrace();
            exit(1);
        }

        Scanner sc = new Scanner(System.in);

        switch (args[0]) {
            case "P1": {
                // use the network constructed based on the specification in args[1]
                String[] query = getQueriedNode(sc);
                String variable = query[0];
                String value = query[1];
                bn.setOrder();
                // execute query of p(variable=value)
                QueryInput ask = new QueryInput(variable, QueryInput.resolveBoolean(value));
                double result = bn.query(ask);
                printResult(result);
            }
            break;

            case "P2": {
                // use the network constructed based on the specification in args[1]
                bn.setOrder();
                String[] query = getQueriedNode(sc);
                String variable = query[0];
                String value = query[1];
                String[] order = getOrder(sc);
                bn.setOrder(order);
                // execute query of p(variable=value) with given order of elimination
                QueryInput ask = new QueryInput(variable, QueryInput.resolveBoolean(value));
                double result = bn.query(ask);
                printResult(result);
            }
            break;

            case "P3": {
                // use the network constructed based on the specification in args[1]
                String[] query = getQueriedNode(sc);
                String variable = query[0];
                String value = query[1];
                ArrayList<String[]> evidence = getEvidence(sc);
                String[] order = TopologicalOrdering.reverseOrder(bn);
                bn.setOrder(order);
                // execute query of p(variable=value|evidence) with an order
                QueryInput ask = new QueryInput(variable, QueryInput.resolveBoolean(value), evidence);
                double result = bn.query(ask);
                printResult(result);
            }
            break;

            case "P4": {
                // use the network constructed based on the specification in args[1]

            }
            break;
        }
        sc.close();
    }

    // method to obtain the evidence from the user
    private static ArrayList<String[]> getEvidence(Scanner sc) {

        System.out.println("Evidence:");
        ArrayList<String[]> evidence = new ArrayList<>();
        String[] line = sc.nextLine().split(" ");

        for (String st : line) {
            st = st.toUpperCase();
            String[] ev = st.split(":");
            evidence.add(ev);
        }
        return evidence;
    }

    // method to obtain the order from the user
    private static String[] getOrder(Scanner sc) {

        System.out.println("Order:");
//        String[] val = sc.nextLine().split(",");
        String[] val = Arrays.stream(sc.nextLine().split(",")).map(String::toUpperCase).toArray(String[]::new);
        return val;
    }

    // method to obtain the queried node from the user
    private static String[] getQueriedNode(Scanner sc) {

        System.out.println("Query:");
//        String[] val = sc.nextLine().split(":");
        String[] val = Arrays.stream(sc.nextLine().split(":")).map(String::toUpperCase).toArray(String[]::new);

        return val;

    }

    // method to format and print the result
    private static void printResult(double result) {

        DecimalFormat dd = new DecimalFormat("#0.00000");
        System.out.println(dd.format(result));
    }

}
