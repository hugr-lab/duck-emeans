package emeansplay

import scala.scalajs.js
import scala.scalajs.js.annotation.JSGlobal

/** Facade for window.EmeansDuck, set by ../playground-bridge.js (an ES module that runs before this script). */
@js.native
@JSGlobal("EmeansDuck")
object EmeansDuck extends js.Object:
  def start(extensionUrl: String): js.Promise[DuckSession] = js.native

@js.native
trait DuckSession extends js.Object:
  val version: String = js.native
  def setPoints(name: String, xy: js.Array[Double]): js.Promise[Unit] = js.native
  def trace(options: js.Object): js.Promise[JsTrace] = js.native
  def traceSql(options: js.Object): String = js.native
  def timeHighDim(n: Int, d: Int, k: Int): js.Promise[JsTiming] = js.native

@js.native
trait JsTrace extends js.Object:
  val batchSize: Double = js.native
  val alpha0: Double = js.native
  val frames: js.Array[JsFrame] = js.native

@js.native
trait JsFrame extends js.Object:
  val epoch: Double = js.native
  val trainMse: Double = js.native
  val testMse: Double = js.native
  val centroids: js.Array[Double] = js.native

@js.native
trait JsTiming extends js.Object:
  val generateMs: Double = js.native
  val clusterMs: Double = js.native
  val batchSize: Double = js.native
  val passes: Double = js.native
  val objective: Double = js.native
  val sql: String = js.native

/** What one emeans_trace call is asked for. */
final case class TraceOptions(k: Int, nepochs: Double, seed: Int, method: String, framesPerEpoch: Double,
                              batchSize: Int, revival: Boolean):
  def toJs: js.Object = js.Dynamic.literal(k = k, nepochs = nepochs, seed = seed, method = method,
    framesPerEpoch = framesPerEpoch, batchSize = batchSize, revival = revival)

object Convert:
  def trace(t: JsTrace): Trace =
    Trace(t.batchSize.toLong, t.alpha0,
      t.frames.toVector.map(f => Frame(f.epoch, f.trainMse, f.testMse, f.centroids.toArray)))
