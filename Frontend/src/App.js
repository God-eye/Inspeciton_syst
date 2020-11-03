import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import { BtnGrp } from "./components/VideoSelectionBtnGrp/BtnGrp";
import { Pg2layout } from "./components/Page2/Pg2layout";

function App() {
  return (
    <div className="App">
      <Router>
        <Switch>
          <Route exact path="/">
            <BtnGrp />
          </Route>
          <Route exact path="/home">
            <Pg2layout />
          </Route>
          <Route exact path="/test"></Route>
        </Switch>
      </Router>
    </div>
  );
}

export default App;
