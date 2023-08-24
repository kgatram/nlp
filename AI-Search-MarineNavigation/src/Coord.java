/********************Starter Code
 * 
 * This represents the coordinate data structure (row, column)
 * and prints the required output
 *
 *
 * @author at258, 220025456
 *   
 */

public class Coord {
	private int r;//row
	private int c;//column

	private String direction;

	private int priority;

	/**
	 * Coord constructor
	 * @param row row
	 * @param column column
	 */
	public Coord(int row,int column) {
		r=row;
		c=column;
		direction = "";
		setPriority();
	}

	/**
	 * Coord constructor
	 * @param row row
	 * @param column column
	 * @param dir direction
	 */
	public Coord(int row, int column, String dir) {
		r=row;
		c=column;
		direction = dir;
		setPriority();
	}

	/**
	 * set priority of this coordinate based on direction
	 */
	public void setPriority() {
		switch (direction) {
		case "Right" -> priority = 4;
		case "Down" -> priority = 3;
		case "Left" -> priority = 2;
		case "Up" -> priority = 1;
		default -> priority = 0;
		}
	}

	/**
	 * get priority of this coordinate
	 * @return priority of this coordinate
	 */
	public int getPriority() {
		return priority;
	}

	/**
	 * print the required output
	 */

	public String toString() {
		return "("+r+","+c+")";
	}

	/**
	 * get direction of this coordinate
	 * @return direction of this coordinate
	 */

	public String getDirection() {
		return direction;
	}

	/**
	 * set direction of this coordinate
	 * @param direction direction of this coordinate
	 */
	public void setDirection(String direction) {
		this.direction = direction;
	}

	/**
	 * get row of this coordinate
	 * @return row of this coordinate
	 */

	public int getR() {
		return r;
	}

	/**
	 * get column of this coordinate
	 * @return column of this coordinate
	 */
	public int getC() {
		return c;
	}

	/**
	 * check if this coordinate is equal to another coordinate
	 * @param o another coordinate
	 * @return true if this coordinate is equal to another coordinate
	 */
	@Override
	public boolean equals(Object o) {

		Coord coord=(Coord) o;
		if(coord.r==r && coord.c==c) {
			return true;
		}
		return false; 

	}

}
