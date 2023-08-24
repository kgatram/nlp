/**
 * This is an executable class to search one or more words in a text file.
 * Reads in a simple text file(utf-8) and searches for one or more words. It presents count-the number of times this word
 * appears in the file, and report the counts to the terminal. Count is reported for all the words keyed is search args.
 * A summary of Total search count is present for more than two search words.
 * Search is ase-sensitive.
 * Usage from CLI: java WordCounter &lt;filename&gt; &lt;searchTerm&gt;
 * 
 * @author  220025456
 * @version "%I%, %G%"
 * @since JDK17
 * 
 */

public class WordCounter {

    /**
     *Main method for word count. get following arguments from command line and passes argument to
     *WordSearchAndCount class for count and print.
     *
     * @param args The first argument is the name (and path) of the file containing the text.
     *             The second argument is the word that is being searched for.
     * 
     */
    public static void main(String[] args) {
        WordSearchAndCount wCount = new WordSearchAndCount();
        wCount.validateInputs(args.length);
        int[] count = wCount.searchCount(args);
        wCount.printCount(args, count);
    }

}
