

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.regex.Pattern;

/**
 * WordSearchAndCount class gets arguments from caller and perform validation, search count and printing results to terminal.
 * validateInputs method provides input argument validation. If either file name or word search or both are missing it will display
 * an error message in the terminal.
 * 
 *searchCount method will search a given word in a file and count the number of its occurence. Search is done case-sensitive.
 *In case file is not found at specified path location it displays an error message on File not found.
 *
 *printCount prints the search count in terminal display. It display a text message for single word search however for
 *multi-word search it displays the count in tabular form with a help of printMultiline method.
 * 
 * @author  220025456
 * @version "%I%, %G%"
 * @since JDK17
 * 
 */

public class WordSearchAndCount {

     private static final int MAX_DISPLAY = 100;

    /**
     * validateInputs method provides input argument validation. if either file name or word search or both are missing it will display
     * an error message in the terminal.
     * @param    no_of_parms  no. of arguments keyed on CLI.
     *
     */
    public void validateInputs(int no_of_parms) {
        if (no_of_parms < 2) {
            //System.out.println("Required arguments not passed.");
            System.out.println("Usage: java WordCounter <filename> <searchTerm>");
            System.exit(1);
        }
    }

    /**
     *searchCount method will search a given word in a file and count the number of its occurence. Search is done case-sensitive.
     *in case file is not found at specified path location it also displays an error message on File not found.
     * @param  args The first argument is the name (and path) of the file containing the text.
     *              The second argument is the word that is being searched for.
     * @return integer array Return count of searched arguments.
     *
     */
    public int[] searchCount(String[] args) {
        File textfile = new File(args[0]);
        Pattern delimiter = Pattern.compile("[\\s\\W]");                            //\\u2014\\u2018\\u2019\\u201C
        int[] count = new int[args.length];

        try {
            for (int i = 1; i < args.length; i++) {

                Scanner src = new Scanner(textfile, "utf-8");
                src.useDelimiter(delimiter);
                Pattern pattern = Pattern.compile(args[i]);

                while (src.hasNext()) {
                    if (src.hasNext(pattern)) {
                        count[i]++;
                    }

                    src.next();
                }

                src.close();
            }

        } catch (FileNotFoundException e) {
            System.out.println("File not found: " + textfile.getName());
            //System.out.println("WordCounter Aborting...");
            System.exit(1);
        }

        return count;

    }


    /**
     *printCount prints the search count in terminal display. It display a text message for single word search however for
     *multi-word search it displays the count in tabular form with a help of printMultiline method.
     * @param   args   The first argument is CLI arguments.
     * @param   count  The second argument is word count.
     *
     *
     */
    public void printCount(String[] args, int[] count) {

        if (args.length == 2) {
            String times = count[1] == 1 ? " time." : " times.";
            System.out.println("The word '" + args[1] + "' appears " + count[1] + times);
        }
        else {
            printMultiline(args, count);
        }
    }

   /**
     *printMultiline prints the search count for more than one word in a tabular form. It also calculate
     *a total for count for all the searches
     * @param   args   The first argument is CLI arguments
     * @param   count  The second argument is word count.
     * 
     *
     */

    private void printMultiline(String[] args, int[] count) {
        int maxlen = 0, minlen = 0;
        int maxdigilen = 0;
        int digilen = 0;
        int total = 0;
        String s = "-";
        String elips = "...";
        String separator, header;

        minlen = "TOTAL".length();
        maxlen = minlen;
        maxdigilen = minlen;

        for (int i = 1; i < args.length; i++) {
                maxlen = Math.max(maxlen, args[i].length());
                maxdigilen = Math.max(maxdigilen, String.valueOf(count[i]).length());
            }

        maxlen = maxlen > MAX_DISPLAY ? MAX_DISPLAY : maxlen;       // adjust for long strings
        separator = "|-" + s.repeat(maxlen) + "-|-" + s.repeat(maxdigilen) + "-|";
        header    = "| " + "WORD " + " ".repeat(maxlen - minlen) + " | " + "COUNT" + " ".repeat(maxdigilen - minlen) + " |";

        for (int i = 1; i < args.length; i++) {

            digilen = String.valueOf(count[i]).length();
            total += count[i];
            args[i] = args[i].length() > maxlen ? args[i].substring(0, (MAX_DISPLAY - elips.length())) + elips : args[i];

            if (i == 1) {
                System.out.println(separator);
                System.out.println(header);
                System.out.println(separator);
            }

            System.out.println("| " + args[i] + " ".repeat(maxlen - args[i].length()) + " | "  + " ".repeat(maxdigilen - digilen) + count[i] + " |");

        }

        // print Total in case of multiple search counts.
        digilen = String.valueOf(total).length();
        System.out.println(separator);
        System.out.println("| " + "TOTAL" + " ".repeat(maxlen - minlen) + " | "  + " ".repeat(maxdigilen - digilen) + total + " |");
        System.out.println(separator);

    }

}
