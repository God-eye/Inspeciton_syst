import React, { useState } from "react";
import { Button } from "../Button/Button";
import { Header } from "../Header/Header";
import { VideoJs } from "../Html5Video/Html5Video";
import { URLContext } from "../../contexts/URLcontext";
import Timer from "../Timer";
import "./pg2layout.css";

export const Pg2layout = () => {
  // anamoly_event = 0
  const [hidden, sethidden] = useState("none");
  const [timerstart, settimerstart] = useState(0);
  const [carasolehide, setcarasolehide] = useState("block");
  const [frezze, setfrezze] = useState(1);

  const hiddenStateChanger = () => {
    settimerstart(1);
  };
  const frezzeswitch = () => {
    setfrezze(0);
  };

  return (
    <URLContext.Consumer>
      {(context) => {
        const { isAnamoly } = context;
        return (
          <div className="maingrid">
            <Header />
            <div className="player">
              <VideoJs color={(isAnamoly === "false")?"green":"red"} />
            </div>

            {/* <Button
              id="greenbutton"
              buttonStyle="btn--red-outline"
              buttonSize="btn--large"
              gridClass="normal"
            >
              Home
            </Button>
            <Button
              id="greenbutton"
              buttonStyle="btn--red-outline"
              buttonSize="btn--large"
              gridClass="anamoly"
            >
              Home
            </Button> */}

            <div className="info_">
              <div className="carasole" style={{ display: carasolehide }}>
                <Button
                  id="enbutton"
                  buttonStyle="btn--red-outline"
                  buttonSize="btn--large"
                  gridClass="rmal"
                  onClick={hiddenStateChanger}
                >
                  start
                </Button>
                {(isAnamoly === "false")?sethidden('none'):sethidden('Block')}
                {/* {(isAnamoly === "false")?settimerstart(0):settimerstart(1)} */}
                {/* {(isAnamoly === "false")?setcarasolehide('Block'):setcarasolehide('none')} */}
              </div>
              <div className="time" style={{ display: hidden }}>
              An anomaly has been detected. An email will be sent to the corresponding authorities in:

                {timerstart ? (
                  <Timer initialMinute={timerstart} frez = {frezze} />
                ) : (
                  <div></div>
                )}
                <Button
                  id="greenbutton"
                  buttonStyle="btn--red-outline"
                  buttonSize="btn--large"
                  gridClass="timerstop"
                  onClick = {frezzeswitch}
                >
                  Disable Email Service
                </Button>
              </div>
            </div>
          </div>
        );
      }}
    </URLContext.Consumer>
  );
};
