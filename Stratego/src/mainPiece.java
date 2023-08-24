import stratego.*;
import stratego.pieces.*;

public class mainPiece {
    public static void main(String[] args) {
        Square land, land2, land3, land4, water, water2;
        Piece sergeant, general, general2;
        Player michael, oz;

        land = new Square(null, 1, 2, false);  // null game, but should still work
        land2 = new Square(null, 1, 3, false);
        land3 = new Square(null, 4, 3, false);
        land4 = new Square(null, 4, 9, false);
        water = new Square(null, 3, 1, true);
        water2 = new Square(null, 4, 1, true);

        michael = new Player("Michael", 0);
        oz = new Player("Ozgur", 1);

        sergeant = new StepMover(michael, land, 4);  // weaker
        general = new StepMover(oz, land3, 9);  // stronger
        general2 = new StepMover(michael, land4, 9);

        System.out.println("Sergeant square = " + sergeant.getSquare().getRow() + sergeant.getSquare().getCol());
        System.out.println("Sergeant owner = " + sergeant.getOwner());
        System.out.println("Sergeant rank = " + sergeant.getRank());

//        sergeant.beCaptured();
//        if (sergeant.getSquare() == null) {
//            System.out.println("Sergeant square = null" );
//        }

//        sergeant.move(land2);
//        System.out.println("Sergeant square = " + sergeant.getSquare().getRow() + sergeant.getSquare().getCol());
//        if (sergeant == land2.getPiece()) {
//            System.out.println("Sergeant = land2 piece" );
//        }
//
//        if (sergeant != land.getPiece()) {
//            System.out.println("Sergeant not = land piece" );
//        }

        System.out.println("General when attacking Sergeant = " + general.resultWhenAttacking(sergeant));
        System.out.println("Sergeant when attacking General = " + sergeant.resultWhenAttacking(general));
        System.out.println("General when attacking General2 = " + general.resultWhenAttacking(general2));

        if (land == sergeant.getSquare()) {
            System.out.println("land = Sergeant" );
        }
        if (land3 == general.getSquare()) {
            System.out.println("land3  = general" );
        }

//        sergeant.attack(land3);
        general.attack(land);

        if (sergeant.getSquare() == null) {
            System.out.println("Sergeant sq null" );
        }
        if (land3.getPiece() == null) {
            System.out.println("land = null" );
        }

        if (land == general.getSquare()) {
            System.out.println("land  = general" );
        }
        System.out.println("general sq " + general.getSquare().getRow() + general.getSquare().getCol() );
        System.out.println("land3 sq " + land3.getRow() + land3.getCol() );
        if (land.getPiece() == general) {
            System.out.println("land = general" );
        }



    }
}
