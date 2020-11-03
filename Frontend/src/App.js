import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import { BtnGrp } from "./components/VideoSelectionBtnGrp/BtnGrp";
import { Pg2layout } from "./components/Page2/Pg2layout";
import NewURLpage from "./components/NewURLpage/NewURLpage";
import URLContextProvider from "./contexts/URLcontext";

function App() {
  return (
    <div className="App">
      <URLContextProvider>
        <Router>
          <Switch>
            <Route exact path="/setvid">
              <BtnGrp />
            </Route>
            <Route exact path="/home">
              <Pg2layout />
            </Route>
            <Route exact path="/">
              <NewURLpage />
            </Route>
          </Switch>
        </Router>
      </URLContextProvider>
    </div>
  );
}

export default App;
