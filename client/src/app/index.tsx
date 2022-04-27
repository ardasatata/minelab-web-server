/**
 *
 * App
 *
 * This component is the skeleton around the actual pages, and should only
 * contain code that should be seen on all pages. (e.g. navigation bar)
 */

import * as React from 'react';
import { Helmet } from 'react-helmet-async';
import { Switch, Route, BrowserRouter } from 'react-router-dom';

import { GlobalStyle } from '../styles/global-styles';

import { HomePage } from './pages/HomePage/Loadable';
import { NotFoundPage } from './pages/NotFoundPage/Loadable';
import { useTranslation } from 'react-i18next';
import { RecordVideo } from './pages/RecordVideo';
import { PlayVideo } from './pages/PlayVideo';
import { FileList } from './pages/FileList';
import { CheckingTool } from './pages/CheckingTool';
import Collection from './pages/Collection';

export function App() {
  const { i18n } = useTranslation();
  return (
    <BrowserRouter>
      <Helmet
        titleTemplate="%s - Minelab"
        defaultTitle="Erhu Trainer | 二胡學習"
        htmlAttributes={{ lang: i18n.language }}
      >
        <meta
          name="description"
          content="An Erhu Trainer|二胡學習 application"
        />
      </Helmet>

      <Switch>
        <Route
          exact
          path={process.env.PUBLIC_URL + '/'}
          component={() => <RecordVideo />}
        />
        <Route
          exact
          path={process.env.PUBLIC_URL + '/list'}
          component={() => <FileList />}
        />
        <Route
          exact
          path={process.env.PUBLIC_URL + '/dev'}
          component={HomePage}
        />
        <Route
          exact
          path={process.env.PUBLIC_URL + '/playback'}
          component={() => <PlayVideo />}
        />
        <Route
          exact
          path={process.env.PUBLIC_URL + '/checking'}
          component={() => <CheckingTool />}
        />
        <Route
          exact
          path={process.env.PUBLIC_URL + '/collection'}
          component={() => <Collection />}
        />
        <Route component={NotFoundPage} />
      </Switch>
      <GlobalStyle />
    </BrowserRouter>
  );
}
