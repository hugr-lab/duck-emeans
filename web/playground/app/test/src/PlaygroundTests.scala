package emeansplay

import utest.*

object PlaygroundTests extends TestSuite:
  private def near(a: Double, b: Double, tol: Double = 1e-12): Boolean = math.abs(a - b) <= tol

  private def frame(epoch: Double, train: Double, test: Double = Double.NaN, c: Array[Double] = Array(0, 0)) =
    Frame(epoch, train, test, c)

  val tests = Tests {
    test("Rng matches mulberry32 in web/demo-core.js") {
      // Reference values printed by the JavaScript implementation (node, mulberry32(7) and mulberry32(1)).
      val r = Rng(7)
      assert(near(r.next(), 0.011704753153026104))
      assert(near(r.next(), 0.061958257574588060))
      assert(near(r.next(), 0.97690763277933002))
      assert(near(Rng(1).gaussian(), 1.4043387784152637))
      assert(near(Rng(0).next(), 0.26642920868471265))
      val b = Rng(3)
      assert((0 until 1000).map(_ => b.below(4)).toSet == Set(0, 1, 2, 3))
    }

    test("every data set is deterministic and stays near the unit square") {
      for ds <- Datasets.all do
        val xy = Datasets.generate(ds, 2000, 5)
        assert(xy.length == 4000)
        assert(xy.sameElements(Datasets.generate(ds, 2000, 5)))
        assert(!xy.sameElements(Datasets.generate(ds, 2000, 6)))
        assert(xy.forall(v => v > -0.25 && v < 1.25))
    }

    test("the data sets have the shapes their names say") {
      // Rings: every point is near radius 0.12, 0.27 or 0.42 from the centre.
      val rings = Datasets.generate(Datasets.rings, 3000, 1)
      val radii = rings.grouped(2).map(p => math.hypot(p(0) - 0.5, p(1) - 0.5)).toSeq
      assert(radii.forall(r => Seq(0.12, 0.27, 0.42).exists(c => math.abs(r - c) < 0.08)))
      // Checkerboard: only the dark squares, where floor of the board row and column have equal parity.
      val board = Datasets.generate(Datasets.checkerboard, 3000, 1)
      val cells = board.grouped(2).map(p => (((p(0) - 0.1) / 0.2).toInt, ((p(1) - 0.1) / 0.2).toInt)).toSet
      assert(cells.forall((i, j) => (i + j) % 2 == 0))
      // The uniform square covers both halves of the box.
      val u = Datasets.generate(Datasets.uniform, 2000, 1).grouped(2).map(_(0)).toSeq
      assert(u.count(_ < 0.5) > 900 && u.count(_ >= 0.5) > 900)
    }

    test("Trace.indexAt picks the last frame at or before an epoch") {
      val t = Trace(8, 0.1, Vector(frame(0, 3), frame(0.5, 2), frame(1, 1), frame(2, 0.5), frame(2, 0.4)))
      assert(t.indexAt(-1) == 0)
      assert(t.indexAt(0.4) == 0)
      assert(t.indexAt(0.5) == 1)
      assert(t.indexAt(1.9) == 2)
      assert(t.indexAt(2) == 4) // the final frame shares its epoch with the last step's
      assert(t.frameAt(5).trainMse == 0.4)
      assert(t.lastEpoch == 2)
      assert(Runs(t, Trace(1, 1, Vector(frame(0, 1), frame(25, 1)))).lastEpoch == 25)
    }

    test("Viewport is a square around the points, y up") {
      val v = Viewport.around(Array(0.0, 0.0, 1.0, 0.5))
      assert(near(v.side, 1.08))
      assert(near(v.minX, -0.04) && near(v.minY, 0.25 - 0.54))
      val (px, py) = v.toPx(0.5, 0.25, 100)
      assert(near(px, 50) && near(py, 50))
      val (x, y) = v.fromPx(px, py, 100)
      assert(near(x, 0.5) && near(y, 0.25))
      assert(v.toPx(v.minX, v.minY, 100) == (0.0, 100.0))
      assert(Viewport.around(Array(1.0, 1.0)).side > 0)
    }

    test("Voronoi labels split the grid between two centroids, with one vertical edge per row") {
      val view = Viewport(0, 0, 1)
      val labels = Voronoi.labels(Array(0.25, 0.5, 0.75, 0.5), view, 4)
      assert(labels.toSeq == Seq(0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1))
      val edges = Voronoi.edges(labels, 4)
      assert(edges == (0 until 4).map(row => (row, 1, true)).toVector)
      // Above and below: horizontal edges between rows 1 and 2.
      val stacked = Voronoi.labels(Array(0.5, 0.75, 0.5, 0.25), view, 4)
      assert(Voronoi.edges(stacked, 4) == (0 until 4).map(col => (1, col, false)).toVector)
    }

    test("the loss scale pads the MSE range and skips missing test scores") {
      val runs = Runs(Trace(1, 1, Vector(frame(0, 10, 12), frame(1, 2, 3))), Trace(1, 1, Vector(frame(0, 4), frame(1, 1))))
      val (lo, hi) = LossScale.yRange(runs)
      // Values 1 to 12: span 11, so 1 - 0.44 and 12 + 0.22.
      assert(near(lo, 0.56) && near(hi, 12.22))
      assert(LossScale.staircase(Seq((0.0, 5.0), (1.0, 3.0), (2.0, 2.0))) ==
        Seq((0.0, 5.0), (1.0, 5.0), (1.0, 3.0), (2.0, 3.0), (2.0, 2.0)))
      assert(LossScale.staircase(Seq.empty).isEmpty)
    }

    test("sizes from the log sliders, and the test set size") {
      assert(Sizes.fromLog(4.5) == 31623)
      assert(Sizes.fromLog(2) == 100)
      assert(Sizes.testSize(2000) == 1000)
      assert(Sizes.testSize(50000) == 10000)
      assert(Sizes.testSize(300000) == 20000)
    }

    test("trace options reach JavaScript with their names") {
      val o = TraceOptions(5, 2.0, 3, "lloyd", 1.0, 0, revival = false).toJs.asInstanceOf[scala.scalajs.js.Dynamic]
      assert(o.k.asInstanceOf[Int] == 5 && o.method.asInstanceOf[String] == "lloyd")
      assert(!o.revival.asInstanceOf[Boolean] && o.framesPerEpoch.asInstanceOf[Double] == 1.0)
    }
  }
