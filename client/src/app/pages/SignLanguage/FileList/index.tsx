// @ts-nocheck
/**
 *
 * RecordVideo
 *
 */
import * as React from 'react';
import styled from 'styled-components/macro';
import { StyleConstants } from '../../../../styles/StyleConstants';
import { PageWrapper } from '../../../components/PageWrapper';
import { Helmet } from 'react-helmet-async';

import { useEffect, useState } from 'react';
import axios from 'axios';
import { Link } from '../../../components/Link';
import { NavBar } from '../NavBar';
import { CheckOutlined, LoadingOutlined } from '@ant-design/icons';
import DataTable from 'react-data-table-component';

import { confirmAlert } from 'react-confirm-alert'; // Import
import 'react-confirm-alert/src/react-confirm-alert.css'; // Import css

import TablePagination from '@material-ui/core/TablePagination';
import IconButton from '@material-ui/core/IconButton';
import FirstPageIcon from '@material-ui/icons/FirstPage';
import KeyboardArrowLeft from '@material-ui/icons/KeyboardArrowLeft';
import KeyboardArrowRight from '@material-ui/icons/KeyboardArrowRight';
import LastPageIcon from '@material-ui/icons/LastPage';
import {PaginationChangePage, PaginationChangeRowsPerPage} from "react-data-table-component/dist/src/DataTable/types";

export function getNumberOfPages(rowCount: number, rowsPerPage: number): number {
	return Math.ceil(rowCount / rowsPerPage);
}

const STORAGE_PAGE_KEY = "SL_PAGE"

interface Props {}

export function FileListSL(props: Props) {
  const [isLoading, setIsloading] = useState(false);

  const [datalist, setDatalist] = useState([]);

  const [page, setPage] = useState<number>(() => {
      // getting stored value
      const saved = localStorage.getItem(STORAGE_PAGE_KEY);
      const initialValue = JSON.parse(saved);
      return initialValue || 1;
  });

  const savePage = (page: number) => {
    localStorage.setItem(STORAGE_PAGE_KEY, page)
    setPage(page)
  }

  const getFileList = async () => {
    setIsloading(true);

    const data = await axios.get('https://140.115.51.243/api/sign-language/file-list');

    console.log(data.data.filepath);
    setDatalist(data.data.filepath);

    setIsloading(false);
  };

  const reset = async () => {
    window.location.reload();
  };

  useEffect(() => {
    getFileList();
    setInterval(() => {
      getFileList();
    }, 120000);
  }, []);

  const requestDelete = async filename => {
    setIsloading(true);

    const data = await axios
      .get(`https://140.115.51.243/api/delete?filename=${filename}`)
      .then(() => reset());

    console.log(data);

    if (data.ok) {
      alert.success('File Deleted Successfully!');
    } else {
      alert.error('Error deleting!, Please try again..');
    }
  };

  const deleteFile = filename => {
    confirmAlert({
      title: 'Warning!',
      message: `Are you sure to delete ${filename} ?`,
      buttons: [
        {
          label: 'Yes',
          onClick: () => requestDelete(filename),
          className: 'bg-red-500',
        },
        {
          label: 'No',
          onClick: () => null,
        },
      ],
    });
  };

  function TablePaginationActions({ count, page, rowsPerPage, onPageChange }) {
    const handleFirstPageButtonClick = () => {
        onPageChange(1);
        savePage(1)
    };

    // RDT uses page index starting at 1, MUI starts at 0
    // i.e. page prop will be off by one here
    const handleBackButtonClick = () => {
        onPageChange(page);
        savePage(page)
    };

    const handleNextButtonClick = () => {
        onPageChange(page + 2);
        savePage(page + 2)
    };

    const handleLastPageButtonClick = () => {
        onPageChange(getNumberOfPages(count, rowsPerPage));
        savePage(getNumberOfPages(count, rowsPerPage))
    };

    return (
        <>
            <IconButton onClick={handleFirstPageButtonClick} disabled={page === 0} aria-label="first page">
                <FirstPageIcon />
            </IconButton>
            <IconButton onClick={handleBackButtonClick} disabled={page === 0} aria-label="previous page">
                <KeyboardArrowLeft />
            </IconButton>
            <IconButton
                onClick={handleNextButtonClick}
                disabled={page >= getNumberOfPages(count, rowsPerPage) - 1}
                aria-label="next page"
            >
                <KeyboardArrowRight />
            </IconButton>
            <IconButton
                onClick={handleLastPageButtonClick}
                disabled={page >= getNumberOfPages(count, rowsPerPage) - 1}
                aria-label="last page"
            >
                <LastPageIcon />
            </IconButton>
        </>
    );
}

    const CustomMaterialPagination = ({ rowsPerPage, rowCount, onChangePage, onChangeRowsPerPage, currentPage }) => (
        <TablePagination
            style={{backgroundColor: 'white', paddingTop: '24px'}}
            component="nav"
            count={rowCount}
            rowsPerPage={rowsPerPage}
            rowsPerPageOptions={[10, 20]}
            page={currentPage - 1}
            // onChangePage={onChangePage}
            // onChangeRowsPerPage={({ target }) => onChangeRowsPerPage(Number(target.value))}
            ActionsComponent={TablePaginationActions}
            onPageChange={onChangePage}
            onRowsPerPageChange={({ target }) => onChangeRowsPerPage(Number(target.value))}
        />
    );

  return (
    <>
      <Helmet>
        <title>List</title>
        <meta
          name="description"
          content="A React Boilerplate application homepage"
        />
      </Helmet>
      <NavBar />
      <PageWrapperMain className={'w-full'}>
        {isLoading ? (
          <div
            className={
              'flex flex-col text-white text-7xl m-auto items-center justify-center'
            }
          >
            <LoadingOutlined className={'mb-12'} />
          </div>
        ) : (
          <div className={'flex h-full w-full bg-black justify-center'}>
            <div className="flex flex-col max-h-screen mt-8 w-full max-w-5xl mt-20">
              <DataTable
                paginationDefaultPage={page}
                noDataComponent={<h1 className={'text-4xl my-12'}>No video record is available</h1>}
                paginationRowsPerPageOptions={[10, 20]}
                pagination={true}
                paginationPerPage={20}
                paginationComponent={CustomMaterialPagination}
                columns={[
                  {
                    name: 'File Name',
                    selector: row => row.filename,
                    style: {
                      fontWeight: 'bold',
                      flex: 1
                    },
                    sortable: true,
                  },
                  {
                    left: true,
                    name: 'Analysis Completed',
                    cell: row => (
                      <div style={{ flex: 1, width: '100%', alignItems: 'center', justifyContent: 'center', alignSelf: 'center', textAlign: 'center'}}>
                        {row.isProcessing ? (
                          <LoadingOutlined />
                        ) : row.isPredictError ? (
                          <div>Error</div>
                        ) : (
                          <CheckOutlined />
                        )}
                      </div>
                    ),
                    style:{
                      flex: 1,
                      width: '100%',
                    }
                  },
                  {
                    style:{
                      flex: 2
                    },
                    name: 'Download or Play Analyzed Video When Completed',
                    cell: row => (
                      <div className="flex flex-row text-sm font-light items-center">
                        {/*{row.isProcessing ? (*/}
                        {/*  <></>*/}
                        {/*) : (*/}
                        {/*  <a*/}
                        {/*    className={*/}
                        {/*      'flex h-full items-center justify-center mr-2 font-black cursor-pointer hover:text-blue-500'*/}
                        {/*    }*/}
                        {/*    href={`https://140.115.51.243/api/download-original?filename=${row.processed}`}*/}
                        {/*  >*/}
                        {/*    {'🗄️ \u00A0 Download Original'}*/}
                        {/*  </a>*/}
                        {/*)}*/}
                        {row.isPredictError === true ? (
                          <></>
                        ) : (
                          <>
                            {/*{row.isProcessing ? (*/}
                            {/*  <></>*/}
                            {/*) : (*/}
                            {/*  <a*/}
                            {/*    className={*/}
                            {/*      'flex h-full items-center justify-center mr-2 font-black cursor-pointer hover:text-blue-500'*/}
                            {/*    }*/}
                            {/*    href={`https://140.115.51.243/api/download-predict?filename=${row.filename}_message_blur.mov`}*/}
                            {/*  >*/}
                            {/*    {'✨ \u00A0 Download Analyzed'}*/}
                            {/*  </a>*/}
                            {/*)}*/}
                            {row.isProcessing ? (
                              <></>
                            ) : (
                              <Link
                                to={
                                  process.env.PUBLIC_URL +
                                  `/sign-language/playback?title=${row.filename}`
                                }
                                className={'text-blue-500'}
                              >
                                <div
                                  className={
                                    'flex h-full items-center justify-center mr-4 font-black cursor-pointer'
                                  }
                                >
                                  {'▶ \u00A0️ Play'}
                                </div>
                              </Link>
                            )}
                          </>
                        )}
                        <a
                          className={
                            'flex h-full items-center justify-center mr-2 font-black cursor-pointer text-red-500 hover:text-blue-500'
                          }
                          onClick={() => deleteFile(row.original)}
                        >
                          {'🚫 \u00A0 Delete'}
                        </a>
                      </div>
                    ),
                  },
                ]}
                data={datalist}
                customStyles={customStyles}
              />
            </div>
          </div>
        )}
      </PageWrapperMain>
    </>
  );
}

const Wrapper = styled.header`
  box-shadow: 0 1px 0 0 ${p => p.theme.borderLight};
  height: ${StyleConstants.NAV_BAR_HEIGHT};
  display: flex;
  position: fixed;
  top: 0;
  width: 100%;
  background-color: ${p => p.theme.background};
  z-index: 2;

  @supports (backdrop-filter: blur(10px)) {
    backdrop-filter: blur(10px);
    background-color: ${p =>
      p.theme.background.replace(
        /rgba?(\(\s*\d+\s*,\s*\d+\s*,\s*\d+)(?:\s*,.+?)?\)/,
        'rgba$1,0.75)',
      )};
  }

  ${PageWrapper} {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
`;

export const PageWrapperMain = styled.div`
  display: flex;
  margin: 0 auto;
  box-sizing: content-box;
  height: calc(100vh - ${StyleConstants.NAV_BAR_HEIGHT});
`;

const customStyles = {
  rows: {
    style: {
      minHeight: '32px', // override the row height
    },
  },
  headCells: {
    style: {
      paddingLeft: '8px', // override the cell padding for head cells
      paddingRight: '8px',
    },
  },
  cells: {
    style: {
      paddingLeft: '8px', // override the cell padding for data cells
      paddingRight: '8px',
    },
  },
};
