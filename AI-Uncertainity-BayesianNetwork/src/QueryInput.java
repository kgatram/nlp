import java.util.ArrayList;
import java.util.List;

/**
 * Query input class encapsulating the query, evidence and query order
 * @author 220025456
 */
public class QueryInput {
    private final String label;
    private final boolean value;
    private final List<QueryInput> evidences = new ArrayList<>();

    /**
     * Constructor to use when no evidence is provided
     *
     * @param label variable
     * @param value true or false
     */
    public QueryInput(String label, boolean value) {
        this(label, value, new ArrayList<>());
    }

    /**
     * Constructor to used when evidence is provided
     *
     * @param label     variable
     * @param value     true or false
     * @param evidences list of evidence
     */
    public QueryInput(String label, boolean value, List<String[]> evidences) {
        this.label = label;
        this.value = value;
        for (String[] evidence : evidences) {
            this.evidences.add(new QueryInput(evidence[0], resolveBoolean(evidence[1])));
        }
    }

    /**
     * Gets the associated label
     *
     * @return label
     */
    public String getLabel() {
        return label;
    }

    /**
     * Gets true or false value of query
     *
     * @return true or false
     */
    public boolean get10() {
        return value;
    }

    /**
     * Gets the list of evidences.
     *
     * @return list of evidence
     */
    public List<QueryInput> getEvidences() {
        return evidences;
    }

    /**
     * Checks if evidence exists
     *
     * @return True if exists
     */
    public boolean hasEvidence() {
        return getEvidences().size() != 0;
    }

    /**
     * check whether query exists in the bayesian network
     *
     * @param network bayesian network
     * @return True if query exists
     */
    public boolean exists(BayesianNetwork network) {
        return network.getNode(label) != null;
    }

    /**
     * converts T or F to boolean
     *
     * @param booleanString T or F  string
     * @return boolean value
     */
    public static boolean resolveBoolean(String booleanString) {
        return booleanString.equalsIgnoreCase("T");
    }

}
