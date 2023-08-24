import java.util.*;
import java.util.stream.Collectors;

/**
 * Bayesian Network Class - Main class for computing the probability of a query based on the evidence provided.
 * The class is also used to validate cpt of all the nodes in the network.
 * @author 220025456
 */

public class BayesianNetwork {
    private final Set<Node> nodes = new LinkedHashSet<>();
    private Set<String> inputOrder = null;

    /**
     * Gets all the nodes of the network.
     *
     * @return nodes
     */
    public Set<Node> getNodes() {
        return nodes;
    }

    /**
     * Adds single node to the network
     *
     * @param label node label
     * @return new node with label
     */
    public Node addNode(String label) {
        Node node = new Node(label);
        this.nodes.add(node);
        return node;
    }

    /**
     * Sets input order of elimination.
     *
     * @param order order of elimination
     */
    public void setOrder(String[] order) {
        this.inputOrder = Arrays.stream(order).collect(Collectors.toCollection(LinkedHashSet::new));

    }

    /**
     * Sets default order of elimination.
     */
    public void setOrder() {
        inputOrder = getNodes().stream().map(Node::getLabel).collect(Collectors.toCollection(LinkedHashSet::new));
    }

    /**
     * Gets a node using a label
     *
     * @param label node label
     * @return node
     */
    public Node getNode(String label) {
        return nodes.stream().filter(x -> x.getLabel().equalsIgnoreCase(label)).findFirst().orElse(null);
    }


    /**
     * Gets a list of factors for a given set of node labels.
     *
     * @param nodeLabels set of node labels.
     * @return list of factors.
     */
    private List<Factor> getFactors(Set<String> nodeLabels) {
        return nodeLabels.stream().map(x -> getNode(x).getCpt().copy()).collect(Collectors.toList());
    }


    /**
     * Calculates the probability of a query for the given evidence.
     *
     * @param queryVar query input object containing the query variable and list of evidence
     * @return probability of the query
     */
    public double query(QueryInput queryVar) {
        if (queryVar.exists(this)) {
            Node qryNode = getNode(queryVar.getLabel());

            Set<String> orderOfElimination = new LinkedHashSet<>(inputOrder);
            Set<String> requiredLabels = getRequiredLabels(queryVar);
            orderOfElimination.retainAll(requiredLabels);
            orderOfElimination.addAll(requiredLabels);

            // add query node to the required labels list.
            requiredLabels.add(qryNode.getLabel());

            // get all the required factors into a list this includes the query node, parents and evidence if any.
            List<Factor> listOfFactors = getFactors(requiredLabels);

            // for evidence node, set non-evidence truth row to 0.
            if (queryVar.hasEvidence()) {
                for (QueryInput evidence : queryVar.getEvidences()) {
                    listOfFactors.forEach(factor -> {
                        Node evidenceNode = getNode(evidence.getLabel());
                        if (factor.includes(evidenceNode)) {
                            factor.zeroised(evidenceNode, !evidence.get10());
                        }
                    });
                }
            }

            for (String eliminateLabel : orderOfElimination) {

                // find all the factors that contains the label
                List<Factor> factorsToJoin = listOfFactors.stream().filter(factor -> factor.includes(getNode(eliminateLabel))).toList();

                // perform join and marginalize algorithm
                Factor resultFactor = factorsToJoin.get(0);
                if (factorsToJoin.size() > 1) {
                    for (int i = 1; i < factorsToJoin.size(); i++) {
                        resultFactor = resultFactor.join(factorsToJoin.get(i));
                    }
                }
                resultFactor = resultFactor.sumOut(getNode(eliminateLabel));
                listOfFactors.removeAll(factorsToJoin);
                listOfFactors.add(resultFactor);
            }

            // join factors if factors are more than one
            if (listOfFactors.size() > 1) {
                Factor resultFactor = listOfFactors.get(0);
                for (int i = 1; i < listOfFactors.size(); i++) {
                    resultFactor = resultFactor.join(listOfFactors.get(i));
                }
                listOfFactors = new ArrayList<>(List.of(resultFactor));
            }

            Factor qryFactor = listOfFactors.get(0);
            // normalize
            qryFactor.normalize();

            // get probability of query variable
            Map<String, Boolean> queryMap = qryFactor.getKeyValue(new boolean[]{queryVar.get10()});
            double proba = qryFactor.getCPTbyKey(queryMap);
            return proba;
        }
        return 0.0;
    }


    /**
     * Keep parents and evidence nodes and remove all others that are not required.
     *
     * @param queryVar query input variable and evidence
     * @return set of required labels
     */
    private Set<String> getRequiredLabels(QueryInput queryVar) {
        Node targetNode = getNode(queryVar.getLabel());
        if (queryVar.hasEvidence()) {
            List<Node> listOfParentNodes = Node.getAllParents(targetNode);
            Set<String> requiredLabelSet = listOfParentNodes.stream().map(Node::getLabel).collect(Collectors.toSet());
            for (QueryInput evidence : queryVar.getEvidences()) {
                Set<String> requiredLabels = getRequiredLabels(evidence);
                requiredLabelSet.addAll(requiredLabels);
                requiredLabelSet.add(evidence.getLabel());
            }
            requiredLabelSet.remove(targetNode.getLabel());
            return requiredLabelSet;

        } else {
            List<Node> listOfParentNodes = Node.getAllParents(targetNode);
            return listOfParentNodes.stream().map(Node::getLabel).collect(Collectors.toSet());
        }
    }


    /**
     * Validates the network by checking that the sum of the given cpt for each node is 1.0
     */
    public void validate() {
        for (Node node : nodes) {
            double total = 0.0;
            Factor nodeCpt = node.getCpt();
            int size = (int) Math.pow(2, nodeCpt.getNodeSet().size());
            List<boolean[]> truthTable = nodeCpt.truthTable();
            for (int i = 0; i < size; i++) {
                  total += nodeCpt.getCpt().get(nodeCpt.getKey(truthTable.get(i)));
                if (i % 2 != 0) {
                        if (total != 1.0) {
                            throw new IllegalArgumentException("Error in cpt for node " + node.getLabel());
                        }
                    total = 0.0;
                }
            }
        }

    }


}
