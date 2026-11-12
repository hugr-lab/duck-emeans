package emeansplay

import com.raquo.laminar.api.L.*
import org.scalajs.dom

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.scalajs.js
import scala.scalajs.js.JSConverters.*
import scala.util.{Failure, Success}

/** The playground: Lloyd and e-means side by side on the same 2-D data, both run by the emeans extension
  * inside DuckDB-WASM (emeans_trace), then replayed frame by frame.
  */
object Main:
  /** A run replays in this many seconds at speed 1. */
  val PlaySeconds = 8.0
  /** e-means frames per epoch; Lloyd records each of its iterations. */
  val FramesPerEpoch = 8.0

  enum Phase:
    case Loading, Ready, Training, Playing, Paused, Finished

  // Settings
  val dataset = Var(0)
  val nLog = Var(4.5)
  val seed = Var(0)
  val kLog = Var(2.0)
  val emeansEpochs = Var(5)
  val lloydEpochs = Var(25)
  val revival = Var(true)
  val speed = Var(1.0)

  // State
  val phase = Var(Phase.Loading)
  val status = Var("Loading DuckDB-WASM and the emeans extension...")
  val points = Var(Array.empty[Double])
  val runs = Var(Option.empty[Runs])
  val playhead = Var(0.0)
  val queries = Var("")
  val timing = Var("Not run yet.")
  val timingSql = Var("")
  private var session: Option[DuckSession] = None

  def n: Int = Sizes.fromLog(nLog.now())
  def k: Int = math.max(2, math.min(Sizes.fromLog(kLog.now()), n / 2))

  //===--------------------------------------------------------------------===//
  // Actions
  //===--------------------------------------------------------------------===//

  def newPoints(): Unit =
    points.set(Datasets.generate(Datasets.all(dataset.now()), n, seed.now()))
    runs.set(None)
    playhead.set(0)
    if phase.now() != Phase.Loading then phase.set(Phase.Ready)

  private def options(method: String): TraceOptions =
    if method == "lloyd" then TraceOptions(k, lloydEpochs.now(), seed.now(), "lloyd", 1, 0, revival = false)
    else TraceOptions(k, emeansEpochs.now(), seed.now(), "emeans", FramesPerEpoch, 0, revival.now())

  private def timed[A](f: => Future[A]): Future[(A, Double)] =
    val start = dom.window.performance.now()
    f.map(a => (a, dom.window.performance.now() - start))

  private def runBoth(s: DuckSession, train: Array[Double]): Future[(Runs, Double, Double)] =
    val test = Datasets.generate(Datasets.all(dataset.now()), Sizes.testSize(n), seed.now() + 1)
    for
      _ <- s.setPoints("train", train.toJSArray).toFuture
      _ <- s.setPoints("test", test.toJSArray).toFuture
      _ = status.set(s"Training Lloyd: ${lloydEpochs.now()} iterations over ${n} points, k = $k ...")
      (l, lMs) <- timed(s.trace(options("lloyd").toJs).toFuture)
      _ = status.set(s"Training e-means: ${emeansEpochs.now()} epochs ...")
      (e, eMs) <- timed(s.trace(options("emeans").toJs).toFuture)
    yield (Runs(Convert.trace(l), Convert.trace(e)), lMs, eMs)

  def train(): Unit = session.foreach { s =>
    newPoints()
    phase.set(Phase.Training)
    runBoth(s, points.now()).onComplete {
      case Success((r, lMs, eMs)) =>
        runs.set(Some(r))
        queries.set(s"${s.traceSql(options("lloyd").toJs)};\n${s.traceSql(options("emeans").toJs)};")
        status.set(f"Trained in DuckDB-WASM: Lloyd in ${lMs / 1000}%.2f s, e-means in ${eMs / 1000}%.2f s " +
          f"(batch size ${r.emeans.batchSize}, alpha0 ${r.emeans.alpha0}%.4f).")
        play(fromStart = true)
      case Failure(e) =>
        status.set(s"Error: ${e.getMessage}")
        phase.set(Phase.Ready)
    }
  }

  private var lastTick = -1.0

  private def frame(): Unit =
    val _ = dom.window.requestAnimationFrame((ts: Double) => tick(ts))

  private def tick(ts: Double): Unit =
    if phase.now() == Phase.Playing then
      runs.now().foreach { r =>
        val dt = if lastTick < 0 then 0.0 else (ts - lastTick) / 1000
        lastTick = ts
        val next = math.min(r.lastEpoch, playhead.now() + dt * r.lastEpoch / PlaySeconds * speed.now())
        playhead.set(next)
        if next >= r.lastEpoch then phase.set(Phase.Finished) else frame()
      }

  def play(fromStart: Boolean): Unit =
    if fromStart then playhead.set(0)
    phase.set(Phase.Playing)
    lastTick = -1
    frame()

  def togglePlay(): Unit = phase.now() match
    case Phase.Playing  => phase.set(Phase.Paused)
    case Phase.Finished => play(fromStart = true)
    case _              => play(fromStart = false)

  def scrub(fraction: Double): Unit = runs.now().foreach { r =>
    phase.set(Phase.Paused)
    playhead.set(fraction * r.lastEpoch)
  }

  def timeHighDim(nv: Int, d: Int, kv: Int): Unit = session.foreach { s =>
    timing.set(s"Building $nv vectors of dimension $d and clustering them into $kv ...")
    s.timeHighDim(nv, d, kv).toFuture.onComplete {
      case Success(t) =>
        timing.set(f"Generating the vectors took ${t.generateMs}%.0f ms; clustering them ${t.clusterMs}%.0f ms " +
          f"(batch size ${t.batchSize}%.0f, ${t.passes}%.0f passes, final objective ${t.objective}%.1f).")
        timingSql.set(t.sql)
      case Failure(e) => timing.set(s"Error: ${e.getMessage}")
    }
  }

  //===--------------------------------------------------------------------===//
  // Drawing
  //===--------------------------------------------------------------------===//

  private var canvases = Map.empty[String, dom.HTMLCanvasElement]

  def redraw(): Unit =
    val xy = points.now()
    if xy.nonEmpty then
      val view = Viewport.around(xy)
      val r = runs.now()
      val at = playhead.now()
      canvases.get("lloyd").foreach(Draw.panel(_, xy, view, r.map(_.lloyd.frameAt(at)), "--lloyd"))
      canvases.get("emeans").foreach(Draw.panel(_, xy, view, r.map(_.emeans.frameAt(at)), "--emeans"))
      canvases.get("loss").foreach(Draw.loss(_, r, at))

  private def canvasFor(name: String, label: String): HtmlElement =
    canvasTag(cls := "plot", aria.label := label, onMountCallback { ctx =>
      canvases += name -> ctx.thisNode.ref
      redraw()
    })

  //===--------------------------------------------------------------------===//
  // Readouts
  //===--------------------------------------------------------------------===//

  private def readout(pick: Runs => Trace): Signal[String] =
    runs.signal.combineWith(playhead.signal).map {
      case (Some(r), at) =>
        val f = pick(r).frameAt(at)
        f"after ${f.epoch}%.2f epochs: train MSE ${f.trainMse}%.5f, test MSE ${f.testMse}%.5f"
      case _ => "not trained yet"
    }

  /** The final test MSEs compared, once the replay reaches the end. */
  val verdict: Signal[String] = runs.signal.combineWith(phase.signal).map {
    case (Some(r), Phase.Finished) =>
      val (l, e) = (r.lloyd.frames.last.testMse, r.emeans.frames.last.testMse)
      val diff = (e - l) / l * 100
      f"e-means after ${r.emeans.lastEpoch}%.0f epochs ends ${math.abs(diff)}%.1f%% ${if diff <= 0 then "below" else "above"} " +
        f"Lloyd after ${r.lloyd.lastEpoch}%.0f iterations in held-out test MSE."
    case _ => ""
  }

  //===--------------------------------------------------------------------===//
  // Controls
  //===--------------------------------------------------------------------===//

  private def slider(labelText: Signal[String], minV: Double, maxV: Double, stepV: Double, initial: Double,
                     onValue: Double => Unit): HtmlElement =
    label(
      span(child.text <-- labelText),
      input(typ := "range", minAttr := minV.toString, maxAttr := maxV.toString, stepAttr := stepV.toString,
        defaultValue := initial.toString, onInput.mapToValue.map(_.toDouble) --> (v => onValue(v))),
    )

  private def dataControls: HtmlElement =
    fieldSet(
      legend("Data"),
      label(span("Distribution"),
        select(onChange.mapToValue.map(_.toInt) --> { i => dataset.set(i); newPoints() },
          Datasets.all.zipWithIndex.map((d, i) => option(value := i.toString, d.name)))),
      slider(nLog.signal.map(v => f"n (training points): ${Sizes.fromLog(v)}%,d"), 3, 5.5, 0.01, nLog.now(),
        v => { nLog.set(v); newPoints() }),
      label(span("Seed"), input(typ := "number", minAttr := "0", stepAttr := "1", defaultValue := "0",
        onInput.mapToValue.map(v => v.toIntOption.getOrElse(0)) --> { v => seed.set(v); newPoints() })),
    )

  private def paramControls: HtmlElement =
    fieldSet(
      legend("Parameters"),
      slider(kLog.signal.map(v => s"Clusters k: ${Sizes.fromLog(v)}"), 0.5, 3, 0.01, kLog.now(), kLog.set),
      slider(emeansEpochs.signal.map(v => s"e-means epochs: $v"), 1, 10, 1, emeansEpochs.now(),
        v => emeansEpochs.set(v.toInt)),
      slider(lloydEpochs.signal.map(v => s"Lloyd iterations: $v"), 1, 50, 1, lloydEpochs.now(),
        v => lloydEpochs.set(v.toInt)),
      label(cls := "check", input(typ := "checkbox", defaultChecked := true, onInput.mapToChecked --> revival),
        span("Dead-cluster revival (e-means)")),
    )

  private val busy: Signal[Boolean] = phase.signal.map(p => p == Phase.Loading || p == Phase.Training)

  private def playControls: HtmlElement =
    div(cls := "play-controls",
      button(typ := "button", cls := "primary", disabled <-- busy, "Train", onClick --> (_ => train())),
      button(typ := "button", disabled <-- runs.signal.map(_.isEmpty).combineWith(busy).map(_ || _),
        child.text <-- phase.signal.map(p => if p == Phase.Playing then "Pause" else "Play"),
        onClick --> (_ => togglePlay())),
      label(cls := "grow", span("Playback"),
        input(typ := "range", minAttr := "0", maxAttr := "1", stepAttr := "0.001",
          aria.label := "Playback position in epochs",
          disabled <-- runs.signal.map(_.isEmpty),
          value <-- runs.signal.combineWith(playhead.signal).map((r, at) => r.fold("0")(x => (at / x.lastEpoch).toString)),
          onInput.mapToValue.map(_.toDouble) --> (v => scrub(v)))),
      slider(speed.signal.map(v => f"Replay speed: $v%.2f"), 0.1, 1, 0.05, speed.now(), speed.set),
    )

  private def panel(title: String, cssClass: String, name: String, read: Runs => Trace): HtmlElement =
    div(cls := "panel",
      h3(cls := cssClass, title),
      canvasFor(name, s"$title: the training points, Voronoi cells and centroids"),
      p(cls := "readout", child.text <-- readout(read)),
    )

  private def embeddingPanel: HtmlElement =
    val (nv, d, kv) = (Var(10000), Var(768), Var(64))
    def choice(labelText: String, v: Var[Int], values: Seq[Int]): HtmlElement =
      label(span(labelText), select(onChange.mapToValue.map(_.toInt) --> v,
        values.map(x => option(value := x.toString, selected := x == v.now(), x.toString))))
    div(
      h2("Embedding-sized vectors"),
      p("Real embeddings have hundreds or thousands of dimensions. This builds n vectors of dimension d around " +
        "16 centres inside DuckDB and clusters them with emeans(), timing both steps. DuckDB-WASM runs on one " +
        "thread, without the native build's threads or Accelerate."),
      div(cls := "row",
        choice("Vectors n", nv, Seq(2000, 10000, 20000)),
        choice("Dimension d", d, Seq(128, 768, 1536)),
        choice("Clusters k", kv, Seq(16, 64, 256)),
        button(typ := "button", disabled <-- busy, "Run", onClick --> (_ => timeHighDim(nv.now(), d.now(), kv.now()))),
      ),
      p(cls := "result", child.text <-- timing.signal),
      pre(cls("empty") <-- timingSql.signal.map(_.isEmpty), child.text <-- timingSql.signal),
    )

  def app: HtmlElement =
    div(
      p(cls := "status", role := "status", child.text <-- status.signal),
      div(cls := "settings", dataControls, paramControls),
      playControls,
      div(cls := "plots",
        panel("Lloyd (full batch)", "lloyd", "lloyd", _.lloyd),
        panel("e-means (batch and step from the scaling laws)", "emeans", "emeans", _.emeans),
        div(cls := "panel",
          h3("Loss (MSE): train vs held-out test"),
          canvasFor("loss", "Training and held-out test MSE of both methods over epochs"),
          p(cls := "readout key",
            span(cls := "lloyd", "Lloyd"), " and ", span(cls := "emeans", "e-means"),
            ": solid is train, dashed is test")),
      ),
      p(cls := "verdict", child.text <-- verdict),
      h2("The queries"),
      p("Each panel is one call of emeans_trace over the tables train and test (one list of points each):"),
      pre(cls("empty") <-- queries.signal.map(_.isEmpty), child.text <-- queries.signal),
      embeddingPanel,
      // Redraw whenever the data, the runs or the playhead change.
      points.signal.combineWith(runs.signal, playhead.signal) --> (_ => redraw()),
    )

  def start(): Unit =
    newPoints()
    render(dom.document.getElementById("app"), app)
    dom.window.addEventListener("resize", (_: dom.Event) => redraw())
    dom.window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (_: dom.Event) => redraw())
    EmeansDuck.start("emeans.duckdb_extension.wasm").toFuture.onComplete {
      case Success(s) =>
        session = Some(s)
        status.set(s"DuckDB ${s.version} with emeans, running in this browser. Press Train.")
        phase.set(Phase.Ready)
      case Failure(e) => status.set(s"Error: ${e.getMessage}")
    }

  def main(args: Array[String]): Unit =
    if dom.document.readyState == dom.DocumentReadyState.loading then
      dom.document.addEventListener("DOMContentLoaded", (_: dom.Event) => start())
    else start()
