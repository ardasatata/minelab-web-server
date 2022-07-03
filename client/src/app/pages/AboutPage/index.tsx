/**
 *
 * About Page
 *
 */
import * as React from 'react';
import styled from 'styled-components/macro';
import { StyleConstants } from '../../../styles/StyleConstants';
import { Helmet } from 'react-helmet-async';
import { NavBar } from '../../components/NavBar';

interface Props {}

export function AboutPage(props: Props) {

  return (
    <>
      <Helmet>
        <title>About Page</title>
        <meta
          name="description"
          content="The development team of this project"
        />
      </Helmet>
      <NavBar />
      <PageWrapperMain>

        <div className={'flex flex-col text-white text-center justify-center w-full'}>
            <h1 className={'text-white text-3xl font-bold'}>
                人工智慧專業小組<br/>
                國立中央大學{`  `}
                <span className={"text-2xl"}>
                   資訊工程系
                </span>
            </h1>
            <ul className={"text-lg"}>
                <li>
                    施國琛 (Timothy K. Shih)
                </li>
                <li>
                    佩馬納 (Aditya Permana)
                </li>
                <li>
                    何迪亞 (Wisnu Aditya)
                </li>
                <li>
                    費群安 (Arda Satata)
                </li>
                <li>
                    赫巴特 (Avirmed Enkhbat)
                </li>
            </ul>

            <h1 className={'text-white text-3xl font-bold'}>
                二胡專業小組<br/>
                國立臺灣藝術大學{`  `}
                <span className={"text-2xl"}>
                    中國音樂學系
                </span>
            </h1>

            <ul className={"text-lg"}>
                <li>
                    林昱廷
                </li>
                <li>
                    林心智
                </li>
                <li>
                    張儷瓊
                </li>
                <li>
                    徐采綺
                </li>
                <li>
                    姜妤璇
                </li>
                <li>
                    林怡君
                </li>
                <li>
                    張盈真
                </li>
            </ul>
        </div>
      </PageWrapperMain>
    </>
  );
}

export const PageWrapperMain = styled.div`
  display: flex;
  margin: 0 auto;
  box-sizing: content-box;
  height: calc(100vh - ${StyleConstants.NAV_BAR_HEIGHT});
`;
