import java.util.*;

/**
 * Topological ordering class. This class is used to create reverse topological ordering for variable elimination.
 * Topological ordering is parent node comes before child node.
 * @author 220025456
 */

public class TopologicalOrdering {

    public static String[] reverseOrder(BayesianNetwork bn) {
        ArrayList<String> ordering = new ArrayList<>();
        Queue<Node> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();

//        get all parent node in the queue
        bn.getNodes().forEach(node -> {
            if (node.getParents().size() == 0) {
                queue.add(node);
                visited.add(node.getLabel());
            }
        });

//        from queue add node to ordering who's all parents are visited.
        while (!queue.isEmpty()) {
            Node parent = queue.poll();
            ordering.add(parent.getLabel());

            parent.getChildren().forEach(child -> {
                boolean allParentsVisited = true;
                for (Node p : child.getParents()) {
                    if (!visited.contains(p.getLabel())) {
                        allParentsVisited = false;
                        break;
                    }
                }
                if (allParentsVisited && !visited.contains(child.getLabel())) {
                    queue.add(child);
                    visited.add(child.getLabel());
                }
            });
        }
        Collections.reverse(ordering);
        return ordering.toArray(String[]::new);
    }

}

