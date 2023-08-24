import java.util.Objects;

/**Starter Code
 * 
 * This is the main class for running the search algorithms. It takes first parameter as the search algorithm to
 * run over configurations ID provided as the second parameter. The additional parameters are optional and are
 * used to run the search algorithm with advanced moves like diagonal moves. To print the map, use the DEBUG
 * parameter as the third parameter.
 * Usage:
 * java A1main <BFS|DFS|BDS|BestF|AStar|EVALUATE> <ConfID> <DEBUG|DIAGONAL>
 *
 * @author at258, 220025456
 *
 */

public class A1main {

	/**
	 * Reads the input and runs the search algorithm on the map
	 *
	 * @param args command line arguments
	 */
	public static void main(String[] args) {

		String algo = args[0];
//		EVALUATE: compare the search algorithm with the configurations
		if (algo.equalsIgnoreCase("evaluate")) {
			SearchComparator comparator = new SearchComparator();
				comparator.compareConfig();
			System.exit(0);
//			DEBUG: print the map
		} else if (args.length > 2 && args[2].equalsIgnoreCase("DEBUG")) {
			Conf conf = Conf.valueOf(args[1]);
			System.out.println("Configuration:"+args[1]);
			System.out.println("Map:");
			printMap(conf.getMap(), conf.getS(), conf.getG());
			System.out.println("Departure port: Start (r_s,c_s): "+conf.getS());
			System.out.println("Destination port: Goal (r_g,c_g): "+conf.getG());
			System.out.println("Search algorithm: "+args[0]);
			System.out.println();
			System.exit(0);
//			run Search algorithm
		} else {
			boolean diagonalMoves = false;
			if (args.length > 1) {
				Conf conf = Conf.valueOf(args[1]);
				// Additional parameters
				if (args.length > 2) {
					diagonalMoves = args[2].equals("DIAGONAL");
				}
				runSearch(algo, conf.getMap(), conf.getS(), conf.getG(), diagonalMoves);
				System.exit(0);
			}
		}
		System.out.println("Usage: java A1main <Algo> <ConfID> <DEBUG|DIAGONAL>");
	}

	/**
	 * Run search algorithm
	 *
	 * @param algo search algorithm to run
	 * @param navigation map provided
	 * @param start start coordinate
	 * @param goal goal coordinate
	 * @param moveDiagonal use advanced moves
	 */
	private static void runSearch(String algo, Map navigation, Coord start, Coord goal, boolean moveDiagonal) {
		Search searchAlgo = algoChecker(algo);
		if (searchAlgo == null) {
			System.out.println("Incorrect search Algorithm input. Expected: BFS|DFS|BDS|BestF|AStar");
			System.exit(0);
		}
		Vessel ferry = new Vessel(start, goal, navigation.getMap(), searchAlgo);
		ferry.setDiagonalMoves(moveDiagonal);
		ferry.navigate();
	}

	/**
	 * Check the search algorithm input
	 *
	 * @param algo search algorithm to run
	 * @return search algorithm
	 */
	private static Search algoChecker(String algo) {
		Search searchAlgo = null;
		switch (algo) {
			case "BFS": //run BFS
				searchAlgo = new BFSearch();
				break;
			case "BDS": // For bidirectional search
				searchAlgo = new BidiSearch();
				break;
			case "DFS": //run DFS
				searchAlgo = new DFSearch();
				break;
			case "BestF": //run BestF
				searchAlgo = new BestFSearch();
				break;
			case "AStar": //run AStar
				searchAlgo = new AStarSearch();
				break;
		}
		return searchAlgo;
	}

	/**
	 * Print the map for debugging
	 *
	 * @param m map to print
	 * @param init start coordinate
	 * @param goal goal coordinate
	 */
	private static void printMap(Map m, Coord init, Coord goal) {

		int[][] map=m.getMap();

		System.out.println();
		int rows=map.length;
		int columns=map[0].length;

		//top row
		System.out.print("  ");
		for(int c=0;c<columns;c++) {
			System.out.print(" "+c);
		}
		System.out.println();
		System.out.print("  ");
		for(int c=0;c<columns;c++) {
			System.out.print(" -");
		}
		System.out.println();

		//print rows 
		for(int r=0;r<rows;r++) {
			boolean right;
			System.out.print(r+"|");
			if(r%2==0) { //even row, starts right [=starts left & flip right]
				right=false;
			}else { //odd row, starts left [=starts right & flip left]
				right=true;
			}
			for(int c=0;c<columns;c++) {
				System.out.print(flip(right));
				if(isCoord(init,r,c)) {
					System.out.print("S");
				}else {
					if(isCoord(goal,r,c)) {
						System.out.print("G");
					}else {
						if(map[r][c]==0){
							System.out.print(".");
						}else{
							System.out.print(map[r][c]);
						}
					}
				}
				right=!right;
			}
			System.out.println(flip(right));
		}
		System.out.println();
	}

	/**
	 * Check if coordinates are the same as current (r,c)
	 *
	 * @param coord coordinate to check
	 * @param r row
	 * @param c column
	 * @return true if coordinates are the same
	 */
	private static boolean isCoord(Coord coord, int r, int c) {
		if(coord.getR()==r && coord.getC()==c) {
			return true;
		}
		return false;
	}

	/**
	 * Flip the triangle edges
	 *
	 * @param right true if right triangle
	 * @return flipped triangle
	 */
	public static String flip(boolean right) {
        //prints triangle edges
		if(right) {
			return "\\"; //right return left
		}else {
			return "/"; //left return right
		}

	}

}
