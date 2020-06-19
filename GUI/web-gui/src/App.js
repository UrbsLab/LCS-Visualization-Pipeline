import React from "react";
import Dashboard from "./Dashboard/Dashboard";
import Splash from "./SplashPage/Splash";

class App extends React.Component {
  render() {
    //const subdomain = "theseus.io"; //Use splash
    const subdomain = "dashboard.theseus.io"; //Use dashboard
    return <div>{subdomain === "theseus.io" ? <Splash /> : <Dashboard />}</div>;
  }
}

export default App;
