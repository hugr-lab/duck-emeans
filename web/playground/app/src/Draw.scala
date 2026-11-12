package emeansplay

import org.scalajs.dom

/** Canvas drawing for the two cluster panels and the loss chart. Colours come from the page's CSS tokens,
  * so light and dark mode follow the stylesheet.
  */
object Draw:
  /** At most this many training points are drawn; the rest only make the panel slower. */
  val MaxDrawnPoints = 15000

  def token(name: String): String =
    dom.window.getComputedStyle(dom.document.documentElement).getPropertyValue(name).trim

  /** Sizes the canvas's backing store to its CSS size and the device pixel ratio; returns the context and
    * the side in device pixels.
    */
  def prepare(canvas: dom.HTMLCanvasElement): (dom.CanvasRenderingContext2D, Double) =
    val size = math.max(1.0, canvas.clientWidth * dom.window.devicePixelRatio)
    canvas.width = size.toInt
    canvas.height = size.toInt
    val ctx = canvas.getContext("2d").asInstanceOf[dom.CanvasRenderingContext2D]
    ctx.clearRect(0, 0, size, size)
    (ctx, size)

  private def drawPoints(ctx: dom.CanvasRenderingContext2D, xy: Array[Double], view: Viewport, size: Double): Unit =
    val n = xy.length / 2
    val stride = math.max(1, n / MaxDrawnPoints)
    val r = math.max(1.0, size / 400)
    ctx.fillStyle = token("--points")
    var i = 0
    while i < n do
      val (px, py) = view.toPx(xy(2 * i), xy(2 * i + 1), size)
      ctx.fillRect(px - r / 2, py - r / 2, r, r)
      i += stride

  private def drawCells(ctx: dom.CanvasRenderingContext2D, frame: Frame, view: Viewport, size: Double,
                        colour: String): Unit =
    val res = if frame.k > 300 then 128 else 192
    val cell = size / res
    ctx.strokeStyle = colour
    ctx.globalAlpha = 0.55
    ctx.lineWidth = math.max(1.0, size / 500)
    ctx.beginPath()
    for (row, col, vertical) <- Voronoi.edges(Voronoi.labels(frame.centroids, view, res), res) do
      if vertical then
        ctx.moveTo((col + 1) * cell, row * cell)
        ctx.lineTo((col + 1) * cell, (row + 1) * cell)
      else
        ctx.moveTo(col * cell, (row + 1) * cell)
        ctx.lineTo((col + 1) * cell, (row + 1) * cell)
    ctx.stroke()
    ctx.globalAlpha = 1

  private def drawCentroids(ctx: dom.CanvasRenderingContext2D, frame: Frame, view: Viewport, size: Double,
                            colour: String): Unit =
    val r = math.max(2.0, size / 160)
    ctx.fillStyle = colour
    for j <- 0 until frame.k do
      val (px, py) = view.toPx(frame.centroids(2 * j), frame.centroids(2 * j + 1), size)
      ctx.beginPath()
      ctx.arc(px, py, r, 0, 2 * math.Pi)
      ctx.fill()

  /** One method's panel: the training points, the Voronoi cells (up to Voronoi.MaxK clusters) and the
    * centroids of the frame.
    */
  def panel(canvas: dom.HTMLCanvasElement, xy: Array[Double], view: Viewport, frame: Option[Frame],
            colourToken: String): Unit =
    val (ctx, size) = prepare(canvas)
    drawPoints(ctx, xy, view, size)
    val colour = token(colourToken)
    frame.foreach { f =>
      if f.k <= Voronoi.MaxK then drawCells(ctx, f, view, size, colour)
      drawCentroids(ctx, f, view, size, colour)
    }

  //===--------------------------------------------------------------------===//
  // The loss chart
  //===--------------------------------------------------------------------===//

  private final case class Plot(ctx: dom.CanvasRenderingContext2D, size: Double, maxEpoch: Double, yLo: Double,
                                yHi: Double):
    val left: Double = size * 0.14
    val bottom: Double = size * 0.88
    val top: Double = size * 0.04
    val right: Double = size * 0.97
    def x(epoch: Double): Double = left + epoch / maxEpoch * (right - left)
    def y(mse: Double): Double = bottom - (mse - yLo) / (yHi - yLo) * (bottom - top)

  private def axes(p: Plot): Unit =
    val ctx = p.ctx
    ctx.strokeStyle = token("--line")
    ctx.fillStyle = token("--muted")
    ctx.lineWidth = 1
    ctx.font = s"${(p.size / 28).round}px system-ui, sans-serif"
    ctx.beginPath()
    ctx.moveTo(p.left, p.top)
    ctx.lineTo(p.left, p.bottom)
    ctx.lineTo(p.right, p.bottom)
    ctx.stroke()
    ctx.textAlign = "center"
    for e <- 0 to p.maxEpoch.ceil.toInt by math.max(1, (p.maxEpoch / 5).ceil.toInt) do
      ctx.fillText(e.toString, p.x(e), p.bottom + p.size / 22)
    ctx.fillText("epochs", (p.left + p.right) / 2, p.size * 0.985)
    ctx.textAlign = "right"
    for v <- Seq(p.yLo, (p.yLo + p.yHi) / 2, p.yHi) do ctx.fillText(f"$v%.3g", p.left - p.size / 80, p.y(v) + 4)

  private def series(p: Plot, points: Seq[(Double, Double)], colour: String, dashed: Boolean): Unit =
    val ctx = p.ctx
    ctx.strokeStyle = colour
    ctx.lineWidth = math.max(1.5, p.size / 220)
    ctx.setLineDash(if dashed then scala.scalajs.js.Array(p.size / 80, p.size / 120) else scala.scalajs.js.Array())
    ctx.beginPath()
    points.headOption.foreach((e, v) => ctx.moveTo(p.x(e), p.y(v)))
    points.drop(1).foreach((e, v) => ctx.lineTo(p.x(e), p.y(v)))
    ctx.stroke()
    ctx.setLineDash(scala.scalajs.js.Array())

  /** Frames up to the playhead, as (epoch, MSE); Lloyd's as a staircase. */
  private def upTo(t: Trace, playhead: Double, score: Frame => Double, stairs: Boolean): Seq[(Double, Double)] =
    val pts = t.frames.take(t.indexAt(playhead) + 1).map(f => (f.epoch, score(f))).filterNot(_._2.isNaN)
    if stairs then LossScale.staircase(pts) else pts

  /** Train (solid) and test (dashed) MSE of both runs up to the playhead, and the playhead itself. */
  def loss(canvas: dom.HTMLCanvasElement, runs: Option[Runs], playhead: Double): Unit =
    val (ctx, size) = prepare(canvas)
    runs.foreach { r =>
      val (lo, hi) = LossScale.yRange(r)
      val p = Plot(ctx, size, math.max(r.lastEpoch, 1e-9), lo, hi)
      axes(p)
      for (t, colour, stairs) <- Seq((r.lloyd, token("--lloyd"), true), (r.emeans, token("--emeans"), false)) do
        series(p, upTo(t, playhead, _.trainMse, stairs), colour, dashed = false)
        series(p, upTo(t, playhead, _.testMse, stairs), colour, dashed = true)
      ctx.strokeStyle = token("--muted")
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(p.x(math.min(playhead, p.maxEpoch)), p.top)
      ctx.lineTo(p.x(math.min(playhead, p.maxEpoch)), p.bottom)
      ctx.stroke()
    }
