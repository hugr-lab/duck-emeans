package emeansplay

/** One recorded moment of a run: centroids are x0 y0 x1 y1 ... */
final case class Frame(epoch: Double, trainMse: Double, testMse: Double, centroids: Array[Double]):
  def k: Int = centroids.length / 2

/** A recorded run of one method (emeans_trace). */
final case class Trace(batchSize: Long, alpha0: Double, frames: Vector[Frame]):
  def lastEpoch: Double = frames.last.epoch

  /** The last frame at or before `epoch`; the first frame before any. Frames are ordered by epoch. */
  def indexAt(epoch: Double): Int =
    var (lo, hi) = (0, frames.length - 1)
    while lo < hi do
      val mid = (lo + hi + 1) / 2
      if frames(mid).epoch <= epoch then lo = mid else hi = mid - 1
    lo

  def frameAt(epoch: Double): Frame = frames(indexAt(epoch))

/** Both runs on the same data. */
final case class Runs(lloyd: Trace, emeans: Trace):
  def lastEpoch: Double = math.max(lloyd.lastEpoch, emeans.lastEpoch)

/** The square of data space the panels show, with pixel mappings (y up). */
final case class Viewport(minX: Double, minY: Double, side: Double):
  def toPx(x: Double, y: Double, size: Double): (Double, Double) =
    ((x - minX) / side * size, size - (y - minY) / side * size)

  def fromPx(px: Double, py: Double, size: Double): (Double, Double) =
    (minX + px / size * side, minY + (size - py) / size * side)

object Viewport:
  /** The smallest square around the points, with a 4% margin on every side. */
  def around(xy: Array[Double]): Viewport =
    val xs = xy.indices.collect { case i if i % 2 == 0 => xy(i) }
    val ys = xy.indices.collect { case i if i % 2 == 1 => xy(i) }
    val (minX, maxX, minY, maxY) = (xs.min, xs.max, ys.min, ys.max)
    val side = math.max(math.max(maxX - minX, maxY - minY), 1e-9) * 1.08
    Viewport((minX + maxX - side) / 2, (minY + maxY - side) / 2, side)

/** Voronoi cells drawn as the edges between grid cells whose nearest centroids differ. */
object Voronoi:
  /** Above this many clusters the cells are too small to show; the e-means playground stops at 700. */
  val MaxK = 700

  /** For a res x res grid over the viewport, row by row from the top, the nearest centroid of each cell's
    * centre.
    */
  def labels(centroids: Array[Double], view: Viewport, res: Int): Array[Int] =
    val k = centroids.length / 2
    val out = new Array[Int](res * res)
    for row <- 0 until res; col <- 0 until res do
      val (x, y) = view.fromPx(col + 0.5, row + 0.5, res)
      var (best, bestD) = (0, Double.MaxValue)
      for j <- 0 until k do
        val (dx, dy) = (x - centroids(2 * j), y - centroids(2 * j + 1))
        val d = dx * dx + dy * dy
        if d < bestD then { best = j; bestD = d }
      out(row * res + col) = best
    out

  /** Grid cells with a different neighbour to the right (vertical edge) or below (horizontal edge), as
    * (row, col, vertical).
    */
  def edges(labels: Array[Int], res: Int): Vector[(Int, Int, Boolean)] =
    val out = Vector.newBuilder[(Int, Int, Boolean)]
    for row <- 0 until res; col <- 0 until res do
      val l = labels(row * res + col)
      if col + 1 < res && labels(row * res + col + 1) != l then out += ((row, col, true))
      if row + 1 < res && labels((row + 1) * res + col) != l then out += ((row, col, false))
    out.result()

/** The loss chart's scales. */
object LossScale:
  /** The MSE range over every frame of both runs, padded 4% below and 2% above as in the e-means
    * playground; NaN test scores are left out.
    */
  def yRange(runs: Runs): (Double, Double) =
    val values = for
      t <- Seq(runs.lloyd, runs.emeans)
      f <- t.frames
      v <- Seq(f.trainMse, f.testMse) if !v.isNaN
    yield v
    val (lo, hi) = (values.min, values.max)
    val span = math.max(hi - lo, 1e-12)
    (lo - 0.04 * span, hi + 0.02 * span)

  /** Steps between consecutive points of a staircase: Lloyd's MSE holds until the next iteration ends. */
  def staircase(points: Seq[(Double, Double)]): Seq[(Double, Double)] =
    points.zip(points.drop(1)).flatMap { case ((x0, y0), (x1, _)) => Seq((x0, y0), (x1, y0)) } ++ points.lastOption

/** The playground's sizes, set by log-scale sliders. */
object Sizes:
  def fromLog(v: Double): Int = math.round(math.pow(10, v)).toInt

  /** The test set: a fifth of the training size, at least 1,000 and at most 20,000 points. */
  def testSize(n: Int): Int = math.max(1000, math.min(20000, n / 5))
