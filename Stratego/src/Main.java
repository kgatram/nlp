import stratego.*;
import stratego.pieces.*;

public class Main {
    public static void main(String[] args) {

//        Player p0 = new Player("Michael", 0);
//        Player p1 = new Player("Oz", 1);
//        Game myGame = new Game(p0, p1);
//
//        if (p0 == myGame.getPlayer(0))
//            System.out.println("p0 true");
//        if (p1 == myGame.getPlayer(1))
//            System.out.println("p1 true");
//
//        //Player p3 = myGame.getPlayer(3);
//
//        Square emptySquare = myGame.getSquare(1,0);
//        if ((emptySquare.getPiece()) == null) {
//            System.out.println("its null");
//        }
//
//        //myGame.getSquare(23, 4);
//
//        Square colonelSquare = myGame.getSquare(0, 2);
//        Piece colonel = new StepMover(p0, colonelSquare, 8);  // rank 8 = colonel
//        Piece presentPiece = colonelSquare.getPiece();
//        if (presentPiece != null) {
//            System.out.println("not null");
//        }
//        if (colonel == presentPiece)
//            System.out.println("colonel == presentPiece");
//
////        PlayerTest

//        Player michael = new Player("Michael", 0);
//        Player oz = new Player("Ozgur", 1);
//
//        // This stuff should work even without a Game object!
//        Square land =  new Square(null, 0, 0, false);
//        Square water = new Square(null, 0, 2, true);
//        Piece captain1 = new StepMover(michael, land, 6);
//        Piece captain2 = new StepMover(michael, land, 6);
//
//
//        System.out.println(michael.getName() + " " + michael.getPlayerNumber());
//        System.out.println(oz.getName() + " " + oz.getPlayerNumber());
//        System.out.println("Michael lost? " + michael.hasLost());
//        michael.loseGame();
//        System.out.println("Michael lost? " + michael.hasLost());
//
////        Test Square
//         Square  land2, land3, water2;
//         Piece lance, general;
//
//        land = new Square(null, 1, 2, false);  // null game, but should still work
//        //land2 = new Square(null, 1, 3, false);
//        //land3 = new Square(null, 0, 3, false);
//        water = new Square(null, 3, 1, true);
//        water2 = new Square(null, 4, 1, true);
//
//        michael = new Player("Michael", 0);
//
//        lance = new Scout(michael, land);
//        //general = new StepMover(michael, land3, 9);
//
//        if (land != null) {
//            System.out.println("land is NOT null");
//            System.out.println("land row = " + land.getRow());
//            System.out.println("land col = " + land.getCol());
//            System.out.println("land game = " + land.getGame());
//            System.out.println("land can be entered = " + land.canBeEntered());
//            //System.out.println("land2 can be entered = " + land2.canBeEntered());
//            System.out.println("water can be entered = " + water.canBeEntered());
//            System.out.println("water2 can be entered = " + water2.canBeEntered());
//            //land.placePiece(general);
//        }
//
//        if (lance == land.getPiece())
//            System.out.println("land.getPiece() = lance");
//
//        land.removePiece();
//        if (land != null) {
//            System.out.println("land is null");
            //land.placePiece(general);
        // }
       // if (general == land.getPiece())
        //    System.out.println("general = land.getPiece()");

        //    Test Bomb

        Player p0 = new Player("Michael", 0);
        Player p1 = new Player("Ozgur", 1);

        Game game = new Game(p0, p1);

        Piece bomb = new Bomb(p0, game.getSquare(0, 3));
        Piece attacker = new StepMover(p1, game.getSquare(1, 3), 7);
        System.out.println("Result when attacking Bomb = " + attacker.resultWhenAttacking(bomb));
//
        attacker.attack(game.getSquare(0, 3));
        if (bomb.getSquare() == null) {
            System.out.println("Bomb is null");
        }

        if (attacker.getSquare() == null) {
            System.out.println("Attacker is null");
        }

        System.out.println("Bomb legal moves = " + bomb.getLegalMoves().size());
        System.out.println("Bomb attack moves = " + bomb.getLegalAttacks().size());

//        Flag Test

//        Player p0 = new Player("Michael", 0);
//        Player p1 = new Player("Ozgur", 1);
//        Game game = new Game(p0, p1);
//        Piece flag = new Flag(p0, game.getSquare(0, 3));
//        Piece attacker = new StepMover(p1, game.getSquare(1, 3), 5);
//
//        System.out.println("Result when attacking Flag = " + attacker.resultWhenAttacking(flag));
//
//        attacker.attack(game.getSquare(0, 3));
//        if (flag.getSquare() == null) {
//            System.out.println("Flag is null");
//        }
//        System.out.println("Flag owner has lost = " + flag.getOwner().hasLost());
//        System.out.println("Game winner = " + game.getWinner());
//
//        System.out.println("Flag legal moves = " + flag.getLegalMoves().size());
//        System.out.println("Flag attack moves = " + flag.getLegalAttacks().size());
//
//        Miner Test
//       ***************

//         Player p0;
//         Player p1;
//         Game game;
//         Piece miner, bomb, sergeant;
//
//        p0 = new Player("Michael", 0);
//        p1 = new Player("Ozgur", 1);
//        game = new Game(p0, p1);
//        miner = new Miner(p0, game.getSquare(0, 3));
//        bomb = new Bomb(p1, game.getSquare(1, 3));
//        sergeant = new StepMover(p1, game.getSquare(0, 2), 4);
//
//        System.out.println("Result when attacking Sergeant = " + miner.resultWhenAttacking(sergeant));
//        System.out.println("Result when attacking Bomb = " + miner.resultWhenAttacking(bomb));
//
//        miner.attack(game.getSquare(1,3)); //attack bomb
//        if (bomb.getSquare() == null) {
//            System.out.println("bomb is null");
//        }
//        System.out.println("Miner square = " + miner.getSquare().getRow() + miner.getSquare().getCol());
//        System.out.println("Miner rank = " + miner.getRank());

    }

}